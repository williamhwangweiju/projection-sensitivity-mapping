#!/usr/bin/env python3
"""
Phase 1: Projection-Sensitivity Profiling
Measure hardware-noise sensitivity of each GPT-2 projection.

python3 experiments/phase1_sensitivity/run_sensitivity_profiling.py \
  --config configs/phase1_quick_test.yaml
"""
import argparse
import yaml
import json
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add local aihwkit-lightning to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "aihwkit-lightning" / "src"))

from src.utils import setup_logger


def _peak_sensitivity(result):
    """Return finite peak absolute delta perplexity for one projection."""
    values = np.asarray(result.get("sensitivities_mean", []), dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.max(np.abs(finite_values)))


def load_config(config_path):
    """Load configuration from YAML file."""
    if not config_path:
        config_path = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(config):
    """Load WikiText dataset using datasets library (better SSL handling)."""
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, Dataset
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset_split = config['dataset'].get('split', 'wikitext-2')
    print(f"Loading {dataset_split} dataset...")
    
    try:
        # Use datasets library (better SSL handling than urllib)
        from datasets import load_dataset as hf_load_dataset
        
        # Map split config to datasets library format
        # Available: wikitext-2-v1, wikitext-103-v1, etc.
        split_config_map = {
            'wikitext-2': 'wikitext-2-v1',
            'wikitext-103': 'wikitext-103-v1',
        }
        
        if dataset_split not in split_config_map:
            raise ValueError(f"Unknown split: {dataset_split}")
        
        # Load from datasets library
        dataset_obj = hf_load_dataset('wikitext', split_config_map[dataset_split], split='test')
        text = '\n'.join(dataset_obj['text'])
        print(f"✅ Loaded {dataset_split}: {len(dataset_obj)} samples")
    except Exception as e:
        print(f"Failed to load {dataset_split} via datasets library: {e}")
        print("Falling back to local synthetic data")
        # Fallback to local synthetic data
        text = "The quick brown fox jumps over the lazy dog. " * 1000

    sequence_length = config['dataset'].get('sequence_length', 128)
    max_tokens = config['dataset'].get('max_tokens', 10000)
    
    # Tokenize in bounded chunks so Hugging Face does not warn about passing
    # one giant >1024-token string through a GPT-2 tokenizer.
    token_ids = []
    chunk_chars = 1000
    for start in range(0, len(text), chunk_chars):
        chunk = text[start:start + chunk_chars]
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        token_ids.extend(ids)
        if len(token_ids) >= max_tokens:
            break

    if not token_ids:
        token_ids = [tokenizer.eos_token_id]

    tokens = torch.tensor(token_ids[:max_tokens], dtype=torch.long)

    class TokenBlockDataset(Dataset):
        """Dataset of fixed-length token blocks."""
        def __init__(self, token_ids, seq_len, pad_token_id):
            self.token_ids = token_ids
            self.seq_len = seq_len
            self.pad_token_id = pad_token_id

        def __len__(self):
            return max(1, int(np.ceil(len(self.token_ids) / self.seq_len)))

        def __getitem__(self, idx):
            start = idx * self.seq_len
            seq = self.token_ids[start:start + self.seq_len]
            if len(seq) < self.seq_len:
                pad = torch.full(
                    (self.seq_len - len(seq),),
                    self.pad_token_id,
                    dtype=seq.dtype
                )
                seq = torch.cat([seq, pad])
            return {"input_ids": seq}

    dataset = TokenBlockDataset(tokens, sequence_length, tokenizer.pad_token_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False
    )
    
    return dataloader, tokenizer


def load_dataset_local(config, tokenizer):
    """Load dataset from local file or use simple test data."""
    from torch.utils.data import DataLoader, Dataset

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    class SimpleDataset(Dataset):
        """Simple dataset for testing."""
        def __init__(self, tokenizer, num_samples=10, seq_len=128):
            self.tokenizer = tokenizer
            self.num_samples = num_samples
            self.seq_len = seq_len
            
            # Use a simple repeated text
            text = "The quick brown fox jumps over the lazy dog. " * 50
            tokens = tokenizer.encode(text, return_tensors='pt')[0]
            self.tokens = tokens
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            start = (idx * 100) % max(1, len(self.tokens) - self.seq_len)
            seq = self.tokens[start:start + self.seq_len]
            if len(seq) < self.seq_len:
                seq = torch.cat([seq, torch.tensor([self.tokenizer.pad_token_id] * (self.seq_len - len(seq)))])
            return {"input_ids": seq}
    
    sequence_length = config['dataset'].get('sequence_length', 128)
    num_samples = max(1, config['dataset']['max_tokens'] // sequence_length)
    dataset = SimpleDataset(tokenizer, num_samples=num_samples, seq_len=sequence_length)
    dataloader = DataLoader(dataset, batch_size=config['dataset']['batch_size'])
    return dataloader


def profile_all_projections(model, tokenizer, dataloader, config, logger):
    """Profile all projections using basic Gaussian noise model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    from src.profilers.sensitivity_profiler import SensitivityProfiler
    profiler = SensitivityProfiler(
        model=model,
        tokenizer=tokenizer,
        device=device,
        seed=config['experiment'].get('seed', 42)
    )
    
    # Get noise settings from config
    noise_std = config['profiling'].get('noise_std_start', 0.023)  # Use start value (Lammie fixed)
    num_seeds = config['profiling'].get('num_seeds', 10)
    
    # Materialize batches once. The profiler computes clean PPL and then
    # reuses the same dataset for each noisy seed; a generator would be
    # exhausted after the clean pass and produce inf noisy PPL.
    profile_batches = []
    for batch in dataloader:
        if isinstance(batch, dict) and 'input_ids' in batch:
            profile_batches.append({"input_ids": batch["input_ids"]})
        else:
            profile_batches.append({"input_ids": batch})

    if not profile_batches:
        raise ValueError("No batches available for sensitivity profiling")
    
    # All 49 projections for GPT-2-small
    blocks = 12
    projections_list = []
    
    for block_idx in range(blocks):
        block_id = f"block_{block_idx}"
        for proj_name in ["q_proj", "out_proj", "fc1", "fc2"]:
            projections_list.append((block_id, proj_name))
    
    # LM head
    projections_list.append(("block_11", "lm_head"))  # Use block_11 as pseudo-block for lm_head
    
    all_results = {}
    logger.info(f"Profiling {len(projections_list)} projections with {num_seeds} seeds each")
    
    pbar = tqdm(total=len(projections_list), desc="Profiling projections")
    
    for block_id, proj_name in projections_list:
        display_name = f"{block_id}/{proj_name}"
        logger.info(f"Profiling {display_name}...")
        
        try:
            result = profiler.profile_projection(
                block_id=block_id if proj_name != "lm_head" else "block_11",
                proj_name=proj_name,
                dataset=profile_batches,
                noise_std=noise_std,
                num_seeds=num_seeds
            )
            
            # Store result
            if block_id not in all_results:
                all_results[block_id] = {}
            
            # Convert to format compatible with analysis script
            all_results[block_id][proj_name] = {
                "ppl_clean": result["ppl_clean"],
                "sensitivities_mean": [result["sensitivity_mean"]],
                "sensitivities_std": [result["sensitivity_std"]],
                "noise_levels": [result["noise_level"]],
                "delta_ppl_mean": [result["sensitivity_mean"]],
                "delta_ppl_std": [result["sensitivity_std"]],
            }
            
            logger.info(f"  {display_name}: ppl_clean={result['ppl_clean']:.2f}, "
                       f"sensitivity={result['sensitivity_mean']:.4f}±{result['sensitivity_std']:.4f}")
        
        except Exception as e:
            logger.warning(f"Failed to profile {display_name}: {e}")
            all_results.setdefault(block_id, {})[proj_name] = {
                "ppl_clean": np.nan,
                "sensitivities_mean": [np.nan],
                "sensitivities_std": [np.nan],
                "noise_levels": [noise_std],
                "delta_ppl_mean": [np.nan],
                "delta_ppl_std": [np.nan],
            }
        
        pbar.update(1)
    
    pbar.close()
    return all_results


def compute_normalized_sensitivities(all_results):
    """Normalize peak delta perplexities across all non-lm_head projections."""
    baseline_sensitivities = []
    
    for block_id in all_results:
        for proj_name in all_results[block_id]:
            if proj_name == "lm_head":
                continue
            peak_sens = _peak_sensitivity(all_results[block_id][proj_name])
            if np.isfinite(peak_sens):
                baseline_sensitivities.append(peak_sens)
    
    max_sensitivity = max(baseline_sensitivities) if baseline_sensitivities else 1.0
    min_sensitivity = min(baseline_sensitivities) if baseline_sensitivities else 0.0
    
    sensitivity_range = max_sensitivity - min_sensitivity + 1e-6
    
    normalized = {}
    for block_id in all_results:
        normalized[block_id] = {}
        for proj_name in all_results[block_id]:
            peak_sens = _peak_sensitivity(all_results[block_id][proj_name])
            norm_value = (peak_sens - min_sensitivity) / sensitivity_range
            normalized[block_id][proj_name] = float(norm_value)
    
    return normalized


def compute_peak_sensitivities(all_results):
    """Return raw peak sensitivities (no normalization)."""
    raw = {}
    for block_id in all_results:
        raw[block_id] = {}
        for proj_name in all_results[block_id]:
            raw[block_id][proj_name] = _peak_sensitivity(all_results[block_id][proj_name])
    return raw


def save_results(all_results, sensitivity_values, config, logger, sensitivity_mode):
    """Save profiling results to JSON."""
    results_dir = Path(__file__).parent.parent.parent / "data" / "results"
    profiles_dir = Path(__file__).parent.parent.parent / "data" / "profiles"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    full_results_file = results_dir / f"phase1_full_results_{timestamp}.json"
    with open(full_results_file, 'w') as f:
        results_serializable = {}
        for block_id in all_results:
            results_serializable[block_id] = {}
            for proj_name in all_results[block_id]:
                result = all_results[block_id][proj_name]
                results_serializable[block_id][proj_name] = {
                    "ppl_clean": float(result["ppl_clean"]),
                    "sensitivities_mean": [float(x) for x in result["sensitivities_mean"]],
                    "sensitivities_std": [float(x) for x in result.get("sensitivities_std", np.zeros_like(result["sensitivities_mean"]))],
                    "noise_levels": [float(x) for x in result["noise_levels"]]
                }
                if result.get("delta_ppl_all") is not None:
                    results_serializable[block_id][proj_name]["delta_ppl_all"] = [
                        [float(x) for x in row]
                        for row in np.asarray(result["delta_ppl_all"])
                    ]
        json.dump(results_serializable, f, indent=2)
    logger.info(f"Full results saved to: {full_results_file}")
    
    # Save sensitivity values
    sensitivity_file = profiles_dir / f"sensitivities_{timestamp}.json"
    with open(sensitivity_file, 'w') as f:
        json.dump(sensitivity_values, f, indent=2)
    logger.info(f"{sensitivity_mode.capitalize()} sensitivities saved to: {sensitivity_file}")
    
    # Save summary
    summary_file = results_dir / f"phase1_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("Phase 1: Projection-Sensitivity Profiling Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Device: cuda\n" if torch.cuda.is_available() else f"Device: cpu\n")
        f.write(f"Profiler: AIHWKit-Lightning\n\n")
        
        f.write(f"Per-Projection Sensitivities ({sensitivity_mode}):\n")
        f.write("-" * 60 + "\n")
        for block_id in sorted(sensitivity_values.keys()):
            for proj_name in sorted(sensitivity_values[block_id].keys()):
                sens = sensitivity_values[block_id][proj_name]
                f.write(f"{block_id}/{proj_name:10s}: {sens:.4f}\n")
    
    logger.info(f"Summary saved to: {summary_file}")
    
    return full_results_file, sensitivity_file, summary_file


def main(config_path=None):
    """Run Phase 1 sensitivity profiling experiment."""
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logger("phase1_sensitivity", level=__import__('logging').INFO)
    
    print("\n" + "=" * 70)
    print("PHASE 1: PROJECTION-SENSITIVITY PROFILING")
    print("=" * 70)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Batch size: {config['dataset']['batch_size']}")
    print(f"Max tokens: {config['dataset']['max_tokens']}")
    print(f"Noise levels: {config['profiling']['noise_levels']}")
    print(f"Noise std: {config['profiling']['noise_std_end']}")
    print(f"Number of seeds: {config['profiling']['num_seeds']}")
    print("=" * 70 + "\n")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config['model']['name']}")
    from transformers import GPT2LMHeadModel, AutoTokenizer
    
    model = GPT2LMHeadModel.from_pretrained(config['model']['name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    try:
        dataloader, _ = load_dataset(config)
        logger.info(f"Loaded {len(dataloader)} batches from remote dataset")
    except Exception as e:
        logger.warning(f"Failed to load remote WikiText-2 dataset: {e}. Using simple test data.")
        dataloader = load_dataset_local(config, tokenizer)
        logger.info(f"Loaded {len(dataloader)} batches from local dataset")
    
    # Profile all projections
    all_results = profile_all_projections(model, tokenizer, dataloader, config, logger)
    
    # Compute sensitivity values in selected mode
    normalize = config.get('profiling', {}).get('normalize', True)
    if normalize:
        logger.info("Computing normalized sensitivities...")
        sensitivity_values = compute_normalized_sensitivities(all_results)
        sensitivity_mode = "normalized"
    else:
        logger.info("Computing raw peak sensitivities (no normalization)...")
        sensitivity_values = compute_peak_sensitivities(all_results)
        sensitivity_mode = "raw"
    
    # Save results
    logger.info("Saving results...")
    full_file, sens_file, summary_file = save_results(
        all_results, sensitivity_values, config, logger, sensitivity_mode
    )
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Full results: {full_file}")
    print(f"Sensitivities: {sens_file}")
    print(f"Summary: {summary_file}")
    print("=" * 70 + "\n")
    
    logger.info("Phase 1 profiling complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Projection-Sensitivity Profiling"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    args = parser.parse_args()
    main(args.config)

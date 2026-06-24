"""Projection-level noise sensitivity profiler for GPT-2."""
import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm


class SensitivityProfiler:
    """Profile hardware-noise sensitivity of GPT-2 projections."""
    
    def __init__(self, model, tokenizer, device="cpu", seed=42):
        """Initialize profiler with a GPT-2 model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sensitivities = {}
        self.seed = seed
        self.ppl_clean_cache = {}  # Cache clean perplexity per dataset
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def profile_projection(self, block_id: str, proj_name: str, 
                          dataset, noise_std_start: float = 0.0,
                          noise_std_end: float = 0.05,
                          num_noise_levels: int = 5,
                          num_seeds: int = 1) -> Dict:
        """
        Profile a single projection by injecting controlled noise.
        
        Args:
            block_id: Transformer block identifier (e.g., "block_0")
            proj_name: Projection name (q_proj, out_proj, fc1, fc2, lm_head)
            dataset: Iterable of batches with 'input_ids' key
            noise_std_start: Minimum relative noise scale (fraction of layer weight std)
            noise_std_end: Maximum relative noise scale (fraction of layer weight std)
            num_noise_levels: Number of noise levels to evaluate
            num_seeds: Number of different random noise realizations
        
        Returns:
            Dictionary with perplexity and sensitivity metrics.
        """
        results = {
            "block_id": block_id,
            "proj_name": proj_name,
            "noise_levels": [],
            "ppl_clean": None,
            "ppl_noisy_mean": [],  # Mean over random seeds
            "ppl_noisy_std": [],   # Std over random seeds
            "sensitivities_mean": [],
            "sensitivities_std": []
        }
        
        self.model.eval()
        with torch.no_grad():
            # Compute clean perplexity (baseline) - use cache if available
            dataset_id = id(dataset)  # Unique identifier for this dataset
            if dataset_id not in self.ppl_clean_cache:
                ppl_clean = self._compute_perplexity(dataset)
                self.ppl_clean_cache[dataset_id] = ppl_clean
            else:
                ppl_clean = self.ppl_clean_cache[dataset_id]
            
            results["ppl_clean"] = ppl_clean
            
            # Inject noise at multiple levels
            noise_levels = np.linspace(noise_std_start, noise_std_end, num_noise_levels)
            for noise_level in noise_levels:
                ppl_noisy_list = []
                sensitivity_list = []
                
                # Run multiple times with different random seeds
                for seed_idx in range(num_seeds):
                    torch.manual_seed(self.seed + seed_idx + 1000)
                    np.random.seed(self.seed + seed_idx + 1000)
                    
                    ppl_noisy = self._compute_perplexity_with_noise(
                        dataset, block_id, proj_name, noise_level
                    )
                    ppl_noisy_list.append(ppl_noisy)
                    # Delta perplexity: absolute difference (Lammie et al. metric)
                    sensitivity = ppl_noisy - ppl_clean
                    sensitivity_list.append(sensitivity)
                
                results["noise_levels"].append(noise_level)
                results["ppl_noisy_mean"].append(np.mean(ppl_noisy_list))
                results["ppl_noisy_std"].append(np.std(ppl_noisy_list))
                results["sensitivities_mean"].append(np.mean(sensitivity_list))
                results["sensitivities_std"].append(np.std(sensitivity_list))
        
        return results
    
    def _compute_perplexity(self, dataset):
        """Compute perplexity on a dataset."""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataset:
                input_ids = batch["input_ids"].to(self.device)
                # Use full sequence length from dataset
                # Typically 1024 for Lammie reproduction
                
                outputs = self.model(input_ids, labels=input_ids)
                total_loss += outputs.loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return float(perplexity)
    
    def _compute_perplexity_with_noise(self, dataset, block_id, proj_name, noise_level):
        """Compute perplexity with injected noise in a specific projection."""
        # lm_head shares weights with token embeddings in GPT-2.
        # Temporarily untie lm_head so output-projection noise does not also
        # perturb input embeddings and inflate sensitivity.
        if proj_name == "lm_head":
            orig_lm_head_weight = self.model.lm_head.weight
            self.model.lm_head.weight = torch.nn.Parameter(orig_lm_head_weight.detach().clone())
            proj = self.model.lm_head
            orig_weight = proj.weight.data.clone()

            try:
                layer_std = torch.clamp(orig_weight.std(unbiased=False), min=1e-8)
                noise = torch.randn_like(orig_weight) * (noise_level * layer_std)
                proj.weight.data = orig_weight + noise
                perplexity = self._compute_perplexity(dataset)
            finally:
                self.model.lm_head.weight = orig_lm_head_weight
            return perplexity

        # Save original weights
        proj = self._get_projection(block_id, proj_name)
        orig_weight = proj.weight.data.clone()

        try:
            # Add Gaussian noise scaled by each layer's weight distribution.
            layer_std = torch.clamp(orig_weight.std(unbiased=False), min=1e-8)
            noise = torch.randn_like(orig_weight) * (noise_level * layer_std)
            proj.weight.data = orig_weight + noise

            perplexity = self._compute_perplexity(dataset)
        finally:
            # Restore original weights
            proj.weight.data = orig_weight
        
        return perplexity
    
    def _get_projection(self, block_id: str, proj_name: str):
        """Retrieve a projection layer by block and name."""
        block_idx = int(block_id.split("_")[1])
        block = self.model.transformer.h[block_idx]
        
        if proj_name == "q_proj":
            return block.attn.c_attn
        elif proj_name == "out_proj":
            return block.attn.c_proj
        elif proj_name == "fc1":
            return block.mlp.c_fc
        elif proj_name == "fc2":
            return block.mlp.c_proj
        elif proj_name == "lm_head":
            return self.model.lm_head
        else:
            raise ValueError(f"Unknown projection: {proj_name}")

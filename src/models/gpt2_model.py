"""GPT-2 model utilities and projection analysis."""
import torch
from transformers import GPT2LMHeadModel


class GPT2Analyzer:
    """Analyzer for GPT-2 projections and structure."""
    
    def __init__(self, model_name="gpt2"):
        """Initialize with a pretrained GPT-2 model."""
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.projections = self._extract_projections()
    
    def _extract_projections(self):
        """Extract all projection layers from GPT-2."""
        projections = {}
        
        num_blocks = len(self.model.transformer.h)
        for block_idx, block in enumerate(self.model.transformer.h):
            block_id = f"block_{block_idx}"
            projections[block_id] = {}
            
            # Attention projections
            projections[block_id]["q_proj"] = block.attn.c_attn
            projections[block_id]["out_proj"] = block.attn.c_proj
            
            # Feed-forward projections\
            projections[block_id]["fc1"] = block.mlp.c_fc
            projections[block_id]["fc2"] = block.mlp.c_proj
            if block_idx == num_blocks - 1:
                projections[block_id]["lm_head"] = self.model.lm_head
        
        return projections
    
    def get_projection_sizes(self):
        """Get size of each projection in bytes."""
        sizes = {}
        for block_id, block_projs in self.projections.items():
            sizes[block_id] = {}
            for proj_name, proj in block_projs.items():
                weight_bytes = proj.weight.numel() * 4  # Assuming float32
                sizes[block_id][proj_name] = weight_bytes
        return sizes
    
    def get_num_blocks(self):
        """Get number of transformer blocks."""
        return len(self.model.transformer.h)
    
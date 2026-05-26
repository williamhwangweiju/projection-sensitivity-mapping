"""
Basic example script from IBM 3D-CiM README.
Run this to understand the full end-to-end pipeline.
"""

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.inference import schedule_execution, fast_trace_decoder
from threedsim.models import DecoderOnlyTransformer
from threedsim.modules import TransformerDecoderLayer
from threedsim.modules.base import (
    assign_acc,
    fill_name_fields,
    make_traceable,
    make_use_linear,
)
from threedsim.mapping import Mapper, MapStrategy, Strategy


# Configure the accelerator.
# NOTE: In the the accelerator class, you need to implement the functions that define the latencies [ns] and energy consumed [nJ] for every high-level operation we support.
config = AcceleratorConfig(
    tiles=100, tiers=1024, tier_shape=(512, 512), kv_caching=False
)

num_sequences = 1
start_len = 1
target_len = 12
num_layers = 3
d_model = 512
d_ff = 4 * d_model
vocab_size = 1024

device = "meta"
# Create the accelerator
acc = Accelerator(config, device=device)
decoder_layer_kwargs = {
    "d_model": d_model,
    "nhead": 8,
    "dim_feedforward": d_ff,
}
embedding_layer_kwargs = {
    "vocab_size": vocab_size,
    "embedding_dim": d_model,
    "max_seq_length": target_len,
}
# Create the model
model = DecoderOnlyTransformer(
    TransformerDecoderLayer,
    num_layers=num_layers,
    decoder_layer_kwargs=decoder_layer_kwargs,
    embedding_layer_kwargs=embedding_layer_kwargs,
    device=device,
)

# Each model layer has access to the accelerator
assign_acc(model, acc)

# Create the mapper
mapper = Mapper(
    accelerator=acc,
    model=model,
    map_strategy=MapStrategy(
        strategy=Strategy.GREEDY_IN_ORDER, split_ffn=True, stack_embedding=True
    ),
)
mapper.map_network()
fill_name_fields(model)
make_traceable(model, is_traceable=True)

# Trace the model
make_use_linear(model, use_linear=True)
fast_traced = fast_trace_decoder(
    model, start_len=start_len, target_len=target_len, bsz=num_sequences
)

# Pipelined execution of the model
(
    execution_time,
    memory,
    peak_memory,
    energy,
    flops,
    energy_breakdown,
    latency_breakdown,
) = schedule_execution(
    fast_traced.graph,
    accelerator=model.accelerator,
    copy_and_cleanup_graph=False,
    communication=True,
)
print(f"Execution took {execution_time} ns")
print(f"Required {peak_memory} bytes of scratchpad memory")
print(f"Spent {energy} nJ of energy")

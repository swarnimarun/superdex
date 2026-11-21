# Benchmarks

The published records cover one RTX 5070 Ti with 16 GB of VRAM, a Ryzen 9 9950X3D and WSL2, with the desktop sharing GPU memory. Benchmarks used one worker and ran serially. Each JSON records its workload, settings, and measurement method. Earlier results use different VRAM caps, placements, or kernel builds.

## Q4_K_M fast profile

These are cold code prompts with greedy 256-token continuations and a 131,072-token FP16 KV slot. The 20k result is the median of two graph-chunk trials. The 65k result is the grouped-attention trial. The near-window result is from the selected live profile.

| Input tokens | Prefill tokens/s | Decode tokens/s |
| ---: | ---: | ---: |
| 20,480 | 837 | 48.0 |
| 65,536 | 764 | 42.7 |
| 130,560 | 651 | 35.7 |

The near-window trial peaked at 13.86 decimal GB on the device, including 11.20 GB of runtime growth. The caps were 15 GB and 13 GB respectively. Cold processing rates here exclude cached tokens. GPU headroom and expert cache capacity can vary on a shared desktop. The quality checks use fixed prompts and greedy outputs, with separate comparisons against the reference checkpoint. [Recorded settings and runs](rtx-5070-ti-final-decode.json) · [Checkpoint quality comparison](rtx-5070-ti-q4-128k.json)

## Reproduce the workloads

Start the [fast profile](../README.md#start-the-endpoint) and leave it as the only model worker. Run from the checkout while the endpoint is idle. Write new results under `.runtime/` for local analysis. The exact hashes in the historical code trials are for the patched source used in that trial.

```bash
mkdir -p .runtime/benchmarks
for tokens in 20480 65536 130560; do
    ./scripts/bench-code-context --tokens "$tokens" --output-tokens 256 \
        > ".runtime/benchmarks/code-$tokens.json"
done
./scripts/probe-context --tokens 131008 --seed 23591 \
    > .runtime/benchmarks/retrieval-131008.json
./scripts/bench-prefill 1024 16384 > .runtime/benchmarks/prefill.json
```

`--tokens` counts input tokens. The prompt and completion together must fit the configured slot. `bench-code-context` builds the prompt from pinned llama.cpp sources and records input and output token hashes, timings and static checks of the generated Python. `probe-context` measures a cold retrieval with a generated catalog. To repeat the paired reference and quantization checks or to use a different VRAM cap, follow the commands in the project README. [Checkpoint comparison](../README.md#reproduce-a-checkpoint-comparison) · [Memory settings](../README.md#set-a-memory-budget-and-measure-speed)

## Result files

| Record | Measurements |
| --- | --- |
| [Final decode](rtx-5070-ti-final-decode.json) | CUDA graph chunk size, grouped attention, transfer overlap and the selected live profile |
| [Graph decode](rtx-5070-ti-graph-decode.json) | GPU routing inside CUDA graphs, fused short convolution, expert capacity and cache reuse |
| [Decode profile](rtx-5070-ti-decode-profile.json) | Paired CUPTI traces for scale transfers, token parity and VRAM capacity estimates |
| [Decode optimizations](rtx-5070-ti-decode-optimizations.json) | Elastic KV, fused controllers, GPU routing and Q4 quality comparisons |
| [Compact attention](rtx-5070-ti-compact.json) | Dense and compact global attention, retrieval, code generation and expert-cache replay |
| [Decode control](rtx-5070-ti-decode.json) | Decode and prefix reuse across context lengths |
| [Prefill](rtx-5070-ti-prefill.json) | Microbatch and expert cache sweeps, packed mask and near-full-window retrieval |
| [Q4 at 128k](rtx-5070-ti-q4-128k.json) | Reference and Q4 throughput and quality, full-window retrieval and client tool calls |
| [GPU experts](rtx-5070-ti-gpu-experts.json) | Reference checkpoint with CUDA experts, placement, pinned RAM and retrieval |
| [128k baseline](rtx-5070-ti-128k.json) | Reference placement, retrieval, memory and endpoint checks |
| [16k baseline](rtx-5070-ti.json) | Initial checkpoint, runtime and CPU/GPU placement comparisons |

# Superdex

Run Inkling 2 Mini on one NVIDIA GPU and use it from Pi, Codex, or an Ollama client. The runtime keeps expert weights in RAM and caches them on the GPU as needed. It handles one request at a time, with up to 131,072 tokens for the prompt and reply combined.

## Run it

Use x86-64 Linux or WSL2 with a CUDA 12.8-compatible driver and 26 GiB of free disk space. The test machine had 30 GiB of host RAM. [Setup requirements](docs/runtime.md#start-the-endpoint)

The Q4_K_M fast profile targets SM120 and was tested on a 16 GB RTX 5070 Ti. Install the CUDA 12.8 toolkit with cuBLAS headers, then run these commands.

```bash
INKLING_VARIANT=q4 ./setup.sh
./scripts/build-compact-fa --native
./scripts/start-fast
```

For the standard CUDA path with the reference checkpoint, run `./setup.sh` followed by `./scripts/service start`. Set `INKLING_VRAM_GB` and `INKLING_CONTEXT` at launch to fit a smaller card. [Memory settings](docs/runtime.md#set-a-memory-budget-and-measure-speed)

Run `./scripts/pi` or `./scripts/codex` with the corresponding CLI installed. The endpoint is `http://127.0.0.1:11434` and the model is `inkling-2-mini:latest`. OpenAI-compatible routes use `/v1`.

## Speed

On the RTX 5070 Ti fast profile, cold code prompts generated 256 tokens each. The VRAM limits were 13 GB of runtime growth and 15 GB total, including the desktop.

| Prompt tokens | Prefill tokens/s | Decode tokens/s |
| ---: | ---: | ---: |
| 20,480 | 837 | 48.0 |
| 65,536 | 764 | 42.7 |
| 130,560 | 651 | 35.7 |

[Runtime guide](docs/runtime.md) · [Benchmark data and commands](benchmarks/README.md)

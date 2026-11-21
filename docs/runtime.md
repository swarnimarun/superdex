# Superdex runtime guide

Serve Inkling 2 Mini locally on a single NVIDIA GPU and connect Pi, Codex, or an Ollama-compatible client. The default checkpoint is the 22.87 GB **Unsloth full-precision reference** in mainline GGUF format. Its filename is `UD-Q8_K_XL`. The released experts already use NVFP4 and MXFP8. The reference preserves the NVFP4 experts and expands the MXFP8 weights to BF16. This deployment serves the fixed three-pass text model. [Publisher’s quantization analysis](https://unsloth.ai/docs/models/inkling-2-mini#quantization-analysis) · [Original model card](https://thinkingmachines.ai/model-card/inkling-2-mini/)

On an RTX 5070 Ti with a shared desktop, the **UD-Q4_K_M fast profile** measured 837 prefill and 48.0 decode tokens/s after a cold 20,480-token code prompt. At 130,560 input tokens it measured 651 prefill and 35.7 decode tokens/s. The profile uses a 131,072-token FP16 context and caps runtime VRAM growth at 13 decimal GB. [Benchmark results and reproduction commands](../benchmarks/README.md)

## Start the endpoint

Use x86-64 Linux or WSL2 with glibc 2.38 and libstdc++ providing `GLIBCXX_3.4.32` or newer. Ubuntu 24.04 meets these requirements for the pinned llama.cpp binaries. Install a CUDA 12.8-compatible NVIDIA driver, nvidia-smi, Python 3.9+, Bash, curl, flock, patch, tar, sha256sum, a C/C++ compiler, CMake and Ninja. Allow at least 26 GiB of disk space for one checkpoint and the build. This deployment was tested with 30 GiB of host RAM. Setup downloads pinned artifacts, verifies SHA-256, and builds the patched host libraries. The baseline CUDA kernels and user-space libraries come from the pinned llama.cpp release.

Choose one setup. To run the reference checkpoint, use the standard launcher.

```bash
./setup.sh
./scripts/service start
```

For the RTX 5070 Ti, the fast profile loads **UD-Q4_K_M** with a 131,072-token slot, FP16 keys and values, and GPU expert routing. Its ceilings are 15 decimal GB across the device, including the desktop, and up to 13 decimal GB of runtime growth. The native build needs a CUDA 12.8 toolkit with cuBLAS headers. Set `INKLING_CUDA_TOOLKIT` if the toolkit is outside the usual install locations.

```bash
INKLING_VARIANT=q4 ./setup.sh
./scripts/build-compact-fa --native
./scripts/start-fast
```

Use `./scripts/service stop` before setting up or launching a different profile. To reuse the fast profile when Pi starts the endpoint, save `fast` in `.runtime/host-profile` and `q4` in `.runtime/host-variant`, one line in each. Use `INKLING_VARIANT=q4xl ./scripts/start-fast` after installing the larger 4-bit checkpoint. [Measured performance](#q4_k_m-expert-cache-and-decode)

The managed launcher waits for readiness and runs the endpoint as a transient systemd user service. The service reapplies placement after a backend failure and survives closing the terminal. Use `./scripts/service status`, `./scripts/service logs` and `./scripts/service stop` to manage it. On systems without a user service manager, run `./scripts/serve` in a dedicated terminal. The default listener is `http://127.0.0.1:11434`, the model ID is `inkling-2-mini:latest`, and the default context is 131,072 tokens, including the reply, with a default maximum prefill microbatch of 512 tokens. The XL profile starts at 1,024 when the device ceiling is set to 15 GB. Auto placement can reduce the physical microbatch to fit. Change the port on this server and its clients with `export INKLING_PORT=11435` if Ollama already occupies the default port.

Run the API check in another terminal in the checkout.

```bash
source scripts/env.sh
curl -sS "$OLLAMA_HOST/api/chat" -d '{
  "model":"inkling-2-mini:latest", "stream":false,
  "messages":[{"role":"user","content":"What is 7 times 8?"}]
}'
./scripts/smoke-api
```

The gateway implements Ollama model discovery, show, ps, version, chat and generate. Chat and generate stream NDJSON by default. The OpenAI-compatible base URL is `$INKLING_BASE_URL`. It exposes model listing, Chat Completions and Responses with streaming and function tools. Model installation happens through `setup.sh`.

## Use Pi and Codex

With the respective CLIs on your PATH, run the wrappers from your project directory using their absolute paths, or from this checkout as shown. They preserve the working directory. The Pi wrapper checks the local endpoint on startup and starts the managed service if needed. Pi stores its local provider configuration in `.runtime/pi-agent`. Codex receives a Responses provider and a local model catalog through command-line overrides. Its shell tools work through function calling. Server-hosted web search, custom tools and namespaced connector tools are outside this backend’s supported Responses subset. Both wrappers read the live context size from the gateway, including when the server started with an override in another shell.

```bash
./scripts/pi -p 'Describe this project in one sentence.'
./scripts/codex exec 'Describe this project in one sentence.'
./scripts/smoke-harnesses
```

The harness check gives each agent a temporary file containing a different random value and verifies its final answer. The value is available only through a tool call. For an interactive session, run either wrapper without arguments.

## Select a checkpoint

The stable endpoint alias serves one checkpoint at a time. A fresh checkout defaults to `reference`. To remember a different checkpoint for Pi autostart in this checkout, write its name on a single line in `.runtime/host-variant`. An explicit `INKLING_VARIANT` overrides that file. Other available variants are `q4xl`, `q4` and `iq3`. Download an additional checkpoint once, then switch the service. The Ollama `/api/show` response identifies the loaded quantization. Pi can keep using its existing provider and model ID after a switch.

```bash
python3 checkpoints.py q4xl
./scripts/service stop
INKLING_VARIANT=q4xl ./scripts/service start
```

Set `INKLING_MODE` to `auto`, `static`, `cache` or `hybrid` when starting the service. A running service keeps the environment it started with. To return to the default, stop the service and run `./scripts/service start` with the default environment.

### Q4 at a 131,072-token window

Measured on the RTX 5070 Ti with a desktop sharing VRAM, a 14 GB total ceiling and FP16 KV. Each speed pair used two cold 1,024-token prompts and 64 generated tokens per prompt. Every routed expert projection ran on CUDA. Both Q4s also kept every attention layer on CUDA. The reference retry kept two global attention passes on CPU.

| Checkpoint | File size | Cold prefill tokens/s | Decode tokens/s | Immediate answers correct |
| --- | ---: | ---: | ---: | ---: |
| Reference | 22.87 GB | 486, 503 | 15.2, 15.5 | 31/48 |
| UD-Q4_K_M | 16.61 GB | 521, 611 | 20.4, 22.0 | 28/48 |
| UD-Q4_K_XL | 17.63 GB | 530, 587 | 19.1, 19.7 | 28/48 |

An independent reference run with four global KV banks in host RAM and attention on CUDA also scored 31/48. Each Q4 lost three correct choices and gained zero against the first reference. The paired exact McNemar test gave p = 0.25. The conservative 95% interval for the accuracy change was −18.9 to +7.7 percentage points on this small suite. Both Q4s passed the four free-generation checks for arithmetic, JSON, Python and tool use. Their lower mean negative log likelihood on the immediate-answer suite indicates they assigned more probability to the correct answer on average, even though they chose it less often. These checks leave uncertainty about the quality of a longer coding session. [Local measurements and comparisons](../benchmarks/rtx-5070-ti-q4-128k.json)

Both Q4s preserve the first 20 layers’ native NVFP4 expert weights. The main difference between them is the last four gate/up banks, which use Q5_K in the M file and Q8_0 in the XL file. Unsloth measured 97.9% and 98.0% next-token agreement respectively with its reference on seven domains. [Publisher’s measurements](https://unsloth.ai/docs/models/inkling-2-mini#quantization-analysis)

#### XL expert cache and host RAM

On the lighter desktop load used for this comparison, a bigger GPU cache increased decode throughput with a single Q4_K_XL worker. All three runs kept the 131,072-token FP16 KV entirely on the GPU and used direct I/O. The 20 NVFP4 layers have the same per-expert weight size as the reference. XL's four tail layers use smaller Q8_0 expert weights. The reference retry had space for 166 physical layer/expert pairs and XL held 301 at its initial setting.

| Prefill microbatch | GPU expert slots | Allocated expert cache | 1k prefill / decode (tokens/s) | 16k prefill / decode (tokens/s) | Small-batch cache hit rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 301 | 1,784 MiB | 692 / 19.6 | 612 / 19.9 | 48.1% |
| 512 | 455 | 2,442 MiB | 700 / 22.2 | 611 / 22.0 | 62.5% |
| 256 | 620 | 3,147 MiB | 460 / 23.3 | 431 / 22.6 | 71.1% |

Each row includes two cold 1,024-token trials and one cold 16,384-token trial with 64 generated tokens each. Hit rates cover the backend's microbatches of up to eight tokens, including warmup. Peak backend RSS was 16.6–17.0 GiB. The 455-slot and 620-slot profiles peaked at 13.53 and 13.46 decimal GB total VRAM, including the desktop. GPU headroom changes with other applications, and the cache setting is an upper bound.

The 455-slot XL profile retrieved the expected label from a cold **131,008-token** prompt. The first token arrived after 452.94 seconds and the stream finished after 453.28 seconds with a normal stop. Prefill averaged **289.3 tokens/s**. The first 16k tokens averaged about 591 tokens/s and the last 16k about 191 as global attention traversed more history. The whole device peaked at 13.50 decimal GB, including 10.30 GB of runtime growth. The single worker peaked at 17.15 GiB of host RSS. Ollama discovery, streaming, and file-tool round trips through Pi and Codex passed on this service. [Full-window test and resource measurements](../benchmarks/rtx-5070-ti-q4-128k.json)

To launch the 455-slot profile on this machine:

```bash
./scripts/service stop
INKLING_VARIANT=q4xl INKLING_HOST_GLOBAL_KV=0 INKLING_HOST_OUTPUT=0 \
  INKLING_UBATCH=512 INKLING_CACHE_MIB=2445 INKLING_RESERVE_MIB=896 ./scripts/service start
```

#### Q4_K_M expert cache and decode

On this RTX 5070 Ti, Q4_K_M reduces the host expert bank from 15,066 to 14,094 MiB. The 20 native NVFP4 layers retain their weights. In the four tail layers, the gate/up projection uses Q5_K and the down projection remains Q8_0. At the same 7 GiB bank budget the allocator can fit 1,559 physical layer/expert pairs for M and 1,458 for XL. Actual allocation depends on desktop usage and the conversation length.

The fast profile keeps the router IDs, slot lookup and cache-miss decisions on the GPU during ordinary single-token decode. A miss copies the exact selected expert from mapped pinned RAM before the matrix multiply. The CUDA kernels group normalization, low-rank projections and the gated merge across the query, key and value controllers. They preserve the reference FP32 merge arithmetic. Global FP16 KV reserves stable virtual addresses for the whole slot and commits device memory as the conversation grows. The single worker uses freed space for experts. With 8 MiB growth extents it backed 768 MiB during the 20k code completions and 4,096 MiB near the end of the window. `/api/ps` reports current KV backing, expert slots per pool, route counters and observed kernels.

On cold code prompts using the same 1,536-token microbatch, the smaller quant generated 256 tokens at **36.1 and 37.1 tokens/s after 20,480 input tokens**, compared with 35.3 tokens/s for XL in its single optimized trial. M held 1,528 slots and XL held 1,405 on the first requests. M received an additional 121 MiB of cache in that comparison. The optimized model reproduced its own conventional Q4_K_M runtime's 128 greedy emissions and all 1,024 corresponding top-eight log probabilities. It scored 30/48 on the local forced-choice suite, compared with 29/48 for XL and 31/48 for the publisher’s reference checkpoint. Unsloth reports 97.9% top-token agreement with its reference for M and 98.0% for XL. [Local trials](../benchmarks/rtx-5070-ti-decode-optimizations.json) · [Publisher’s quantization analysis](https://unsloth.ai/docs/models/inkling-2-mini#quantization-analysis)

At 130,560 input tokens, global KV occupied its full 4 GiB. Increasing `INKLING_DECODE_CACHE_HEADROOM_MIB` from 256 to 768 raised GPU expert capacity from 858 to 970 slots and decode from **26.1 to 27.7 tokens/s** on the same 256 generated tokens. Cold code prefill was 649 and 651 tokens/s. The fast profile uses 768 MiB and clips each allocation against the available device budget. At 20,480 input tokens this configuration held **1,639** expert slots and generated a 768-token Python answer at **38.9 tokens/s**, with **834 tokens/s** cold prefill. The generated function passed 1,007 interval tests. The model retrieved the exact value from 131,008 cold tokens at **690 tokens/s**, with the first output at 190 seconds. Whole-device VRAM peaked at 14.62 decimal GB including the desktop, with 11.99 GB of growth. Pi and Codex both read and returned a fresh file value through their tool interfaces.

The expert placement rule leaves immutable scales with their layer. Previously the broad host override included scales, forcing 88 separate 512-byte uploads on every generated token. A paired CUDA activity trace measured roughly 9–12 ms of GPU gaps following those copies. The fix removed all 88 transfers and preserved the same output tokens in all three paired tests. The traced decode rates reached **43.6**, **38.3**, and **31.4 tokens/s** after cold prompts of 20,480, 65,536, and 130,560 tokens. [Paired CUDA activity and throughput](../benchmarks/rtx-5070-ti-decode-profile.json)

The fast profile puts the slot resolver and expert gathers inside the CUDA graph for ordinary single-token decode. At 20,480 input tokens the trace showed 22 graph segments per token, down from 49. The live service generated 768 tokens at 51.0 tokens/s. Near the end of the window, profiled decode reached 32.3 tokens/s with the same output tokens. Set `INKLING_GRAPH_ROUTES=0` to dispatch routing separately from the graph. `scripts/check-graph-routes` verifies captured replay with changing IDs, cache evictions and both expert projection banks. Run it with the model worker stopped. The optional `INKLING_FUSED_SHORTCONV=1` kernel combines four-tap short convolutions with their FP32 residual arithmetic. It matched the reference bit-for-bit and showed similar 20k throughput in the initial pair. [CUDA graph benchmark](../benchmarks/rtx-5070-ti-graph-decode.json)

The fast profile uses 1,024-node CUDA graph chunks and shares global attention K/V loads across three query heads from 65,536 KV rows onward. Set `GGML_CUDA_GRAPH_CHUNK_SIZE=512` or `INKLING_GQA_MIN_K=0` to compare the prior paths. On the 65,536-token code request, grouped attention raised decode from 41.3 to 42.7 tokens/s. A separate overlapped expert-transfer experiment reached 42.3 tokens/s. Each generated the same 256 tokens as the control using 1,308 expert slots. The selected live profile processed 130,560 input tokens at 651 tokens/s and generated the same 256-token output as the previous serving build at 35.7 tokens/s. [Final decode benchmarks](../benchmarks/rtx-5070-ti-final-decode.json)

With `INKLING_DECODE_CACHE_HEADROOM_MIB=1536` and `INKLING_DECODE_CACHE_EXTRA_MIB=5241`, a separate test kept 1,639 slots through 45,056 input tokens and 256 generated tokens at 44.2 tokens/s. The worker committed 1,536 MiB for global FP16 KV, enough for 49,152 rows, and peaked at 13.70 decimal GB across the device. The upper bound under that test's 12 GB growth limit is roughly 53k FP16 rows at that cache size, before driver overhead. The fast profile uses the smaller headroom setting for changing desktop loads.

Set `INKLING_DEVICE_ROUTES=0` to use host-side routing decisions, `INKLING_CONTROLLER_FUSION=0` for the original controller graph, or `INKLING_ELASTIC_KV=0` for fully allocated FP16 KV. Every expert projection still runs on CUDA. Automatic sparse KV and the custom routing and controller kernels target SM120. Other devices retain their supported backend paths. Run `scripts/check-controller` and `scripts/check-device-routes` with the inference worker stopped to verify the compiled kernels. [SGLang’s controller design](https://www.sglang.io/blog/inkling-2-small-day0-support) · [vLLM’s reference arithmetic](https://vllm.ai/blog/2026-11-20-inkling2-mini) · [vAttention](https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/)

#### Cold prefill on the RTX 5070 Ti

At a 15 GB whole-device ceiling, 1,024-token microbatches improved cold 16,384-token prefill from 613 to 745 tokens/s. Both trials used the original mask implementation, a 1,024-token logical batch, 362 expert-cache slots and FP16 KV. A 768-token microbatch reached 605 tokens/s. Holding the microbatch at 512 and spending the extra VRAM on experts gave 614, 619 and 575 tokens/s at ceilings of 14, 14.5 and 15 GB respectively. These are single 16k observations on a shared desktop.

The XL launcher constructs relative-position indices directly and folds causal masking into the learned global bias lookup. The existing CUDA Flash Attention path consumes the resulting FP16 mask. With a 1,024-token microbatch and a 2,048 MiB expert-cache budget, this change raised cold 32,768-token retrieval from 608 to 656 tokens/s. The expected label, the 16 greedy output tokens on a separate 8k prompt, and their reported log probabilities matched at that batch size. To reproduce the earlier mask path, set both `INKLING_FAST_REL_POS=0` and `INKLING_PACK_GLOBAL_MASK=0` when starting the service. [Serial benchmark results](../benchmarks/rtx-5070-ti-prefill.json)

The earlier packed-mask profile requested a 1,536-token batch, 1,037 MiB of expert cache and a 15 GB whole-device ceiling. It reached 686 tokens/s on cold 32k retrieval. A separate 16k request processed 783 prompt tokens/s and generated 64 tokens at 19.3 tokens/s. A 1,024-token batch left more room for resident experts. Both kept FP16 KV on the GPU under the 12 GB incremental cap.

On the same 131,008-token retrieval, the 1,536-token profile evaluated the full prompt at **390.7 tokens/s** in **335.34 s**, compared with 289.3 tokens/s in 452.82 s using the earlier 512-token profile. The last approximately 16k tokens ran at 251 versus 191 tokens/s. Both recovered the exact label and streamed a normal stop. The six-token replies ran at 5.9 versus 14.8 tokens/s respectively. Peak usage with the faster prefill was 14.50 decimal GB total VRAM and 11.43 GB of growth. The one model worker peaked at 17.12 GiB RSS, and the system had at least 11.41 GiB of host RAM available throughout the sampled run. [Retrieval, stage timings and measured memory](../benchmarks/rtx-5070-ti-prefill.json)

Repeating the 16k benchmark through `scripts/bench-prefill` on the final service gave 819 prompt tokens/s and 19.8 generated tokens/s over 64 output tokens. Ollama streaming and native file-tool round trips in Pi and Codex passed. On the same 48 forced-choice questions, the final XL placement scored 29/48, the earlier XL placement scored 28/48, and the reference scored 31/48. The two XL runs chose the same answer on 42 questions. These observations cover the local suite.

We also tested rounding the gathered bias directly into FP16 with a native CUDA kernel. At batch 1,024 it reduced the computed scratch allocation from 2,452 to 2,068 MiB, but cold 32k throughput fell from 656 to 616 tokens/s. At batch 1,536 it fell from 686 to 634. Both implementations exceeded the placement budget at batch 2,048 on this desktop. The packed-mask fallback uses the original FP32 gather.

Run a cold-prompt sweep against the current server with `./scripts/bench-prefill --repeats 2 --decode-tokens 64 1024 16384 32768`. It sends the same input token IDs for each configuration and checks that the backend actually evaluated the entire prompt. For a real retrieval at near-full context use `./scripts/probe-context --tokens 131008 --seed 23591`. Run the commands consecutively on an idle endpoint.

#### SM120 compact global attention and decode cache

For the measured RTX 5070 Ti setup, build the Q4 XL weights and native kernel once. The kernel build needs CUDA 12.8 `nvcc`, cuBLAS development headers and a compatible C++ compiler. Set `INKLING_CUDA_TOOLKIT` or `CUDA_HOME` for a toolkit outside PATH. A system with the pinned base runtime and checkpoint already installed can start at the second command.

```bash
INKLING_VARIANT=q4xl ./setup.sh
./scripts/build-compact-fa --native
./scripts/start-xl-prefill
```

Stop an existing service with `./scripts/service stop` before switching profiles. The launcher keeps one GGUF in pinned host RAM, performs the expert multiplications on CUDA, and keeps the 131,072-token slot's keys and values in FP16. For each of the eight global attention passes, the CUDA kernel reads the learned bias from a compact table. It uses actual KV-slot positions for causal visibility, rounds the bias to FP16 and includes the attention sink. The GPU reads the keys and values at every visible position. Other GPU architectures, host-backed global KV and multi-sequence contexts use the dense path. The launcher checks tensor layout before dispatch. `/api/ps` reports the requested kernel and the compact kernels observed after execution. The RTX 3060 uses the dense path.

On the 5070 Ti, matching the dense kernel's query tile sizes gave identical top-32 token IDs and log probabilities across seven fixed prompts from 64 to 20,480 tokens. It processed a cold 20,480-token retrieval at 930 tokens/s. Whole-query mode reached 957 on a separate run. It agreed on the seven top tokens, with changes in the probabilities. Set `INKLING_NATIVE_FA_MATCH_TILES=0` to opt into that tradeoff. The managed profile recovered the exact label at 20,480 cold tokens at 918.2 tokens/s and at 131,008 tokens at 701.1 tokens/s. At full length the first streamed token arrived after 187.0 seconds. A separate whole-query run recovered the same full-window label in 170.87 seconds of prefill, or 766.7 tokens/s. Its aggressive decode-cache setting subsequently tripped the GPU watchdog and is excluded from the launch profile. [Cold retrieval, token scores and code benchmarks](../benchmarks/rtx-5070-ti-compact.json)

During longer replies, the runtime releases the large prefill workspace and uses that VRAM for additional copies of frequently used experts. It sizes the decode workspace against the entire slot and restores the prefill workspace when the next large prompt arrives. The production transition starts after 12 consecutive decode calls. The single-token CUDA kernel also reads the compact bias directly. On 256-token code completions, it raised decode from 27.1 to 27.5 tokens/s after 20,480 input tokens and from 22.1 to 23.5 after 130,560. Both outputs matched the control byte for byte. On the 48-item forced-choice suite all 192 choice probabilities matched the prior Q4 XL build. The Q4 XL scored 29/48 and the reference scored 31/48. A separate 32,768-token slot fitted a 6,709 MiB decode cache and delivered 30.0 tokens/s after 20,480 source tokens. Use `INKLING_CONTEXT=32768 ./scripts/start-xl-prefill` when this smaller window suits the request. `INKLING_DECODE_CACHE_EXTRA_MIB` bounds the additional cache and `INKLING_DECODE_CACHE_HEADROOM_MIB` bounds the amount it can borrow from the existing allocation allowance. The service reports active cache size, slots, and memory peaks through `/api/ps`. The 15 decimal GB device limit and 13 decimal GB runtime-growth limit apply to both phases.

For an optional single-model code-completion experiment, stop the service and launch with `INKLING_PROMPT_LOOKUP=ngram-simple ./scripts/start-xl-prefill`. The target verifies candidate tokens found in prior text. In three paired 160-token tests this changed throughput from 22.3 to 31.4, 24.2 to 27.5, and 25.1 to 23.9 tokens/s. The first two outputs matched token for token. The diff-parser output diverged after 45 tokens. Ordinary decoding is the default. [Code prompts, acceptance and output comparisons](../benchmarks/rtx-5070-ti-compact.json)

Run the same cold retrievals and a code completion against the live model with `./scripts/bench-compact`. Use `./scripts/bench-code-context --tokens 130560 --output-tokens 256` to measure sustained decode near the end of the window. Inputs and generated tokens share the slot. All tests send requests serially. [Long-context decode control](../benchmarks/rtx-5070-ti-decode.json)

To repeat the paired quality check, run the first command with the reference live and the second after launching a Q4. Use the underlying llama-server port, which defaults to 18567.

```bash
./scripts/compare-quality run --port 18567 --label reference --output .runtime/qa-reference.json
./scripts/compare-quality run --port 18567 --label q4 --output .runtime/qa-q4.json
./scripts/compare-quality compare .runtime/qa-reference.json .runtime/qa-q4.json
```

## Set a memory budget and measure speed

```bash
INKLING_VRAM_GB=10 INKLING_CONTEXT=8192 ./scripts/serve
python3 inkling.py tune --device-gb 10 --context 8192 --ubatch 64 --tokens 96 --repeats 2
./scripts/serve --mode static
```

Stop the running endpoint before benchmarking. Set `INKLING_GPU` to the nvidia-smi device index and `INKLING_VRAM_GB` to the **total device ceiling in decimal GB**. The default is 14 GB. Set `INKLING_VRAM_GB=15` to opt into a 15 GB ceiling on a larger card. Runtime growth is independently capped at 13 decimal GB and may be lower when other applications use GPU memory. The launcher subtracts existing desktop/application usage, estimates model, KV, compute and expert-cache buffers and reserves space for driver overhead and desktop activity. Hosted auto placement uses 768 MiB for a sufficiently large GPU budget with GPU attention and output, 1,536 MiB for a host output projection, and 1,280 MiB for ordinary fallback placement. Set `INKLING_RESERVE_MIB` to override the reserve and `INKLING_CACHE_MIB` to limit the expert cache. It caps standard CUDA buffers before allocation and samples actual whole-device usage throughout startup and inference. A sampled overrun stops the backend. Driver allocations and other processes can change between samples. The lower-level `--vram-gib` flag can additionally limit the model’s incremental budget.

Every indexed expert matrix multiplication runs on the selected CUDA device. During graph scheduling the runtime pins these operations to CUDA, verifies every execution split, and records the first prefill and decode graphs in the backend log. A device or kernel that cannot execute an expert fails explicitly. On the reference checkpoint the middle layers repeat, giving 96 routed projections per token. The 18,711 MiB of expert banks exceed the 13 GB runtime allowance, so cache misses transfer the selected experts from pinned RAM over PCIe. The hosted default uses direct I/O (`--load-mode dio`) to read into the host weight buffers without keeping a second large copy in the Linux file cache. A filesystem that cannot service direct reads falls back to buffered loading. The effective mode is visible in `/api/ps`. Use `INKLING_LOAD_MODE=mmap` on systems where allocating pinned RAM is impractical. [NVIDIA CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory)

`cache` keeps the expert banks in RAM and sizes a GPU cache. `hybrid` also keeps repeated layers on GPU. `static` uses fixed resident banks and streams other routed weights to CUDA. `auto` tries full residency, then measures the actual cache and transfer workspace at progressively smaller microbatches. The GPU memory guard remains active throughout inference. Prompt batching amortizes expert transfers across tokens. Single-stream decode selects routes again for each generated token.

Hosted launches keep one active slot and disable the cross-conversation prompt RAM cache and intermediate context snapshots. A sequential append can reuse the active prefix. Editing history may require another prefill. To re-enable saved prompts or snapshots, set `INKLING_CACHE_RAM_MIB` or `INKLING_CTX_CHECKPOINTS`. The model files for other variants stay on disk. `/api/ps` reports backend RSS, peak RSS and available system RAM in MiB.

The launcher reserves one model worker per Linux user across checkouts and GPU indices. The native process holds the reservation until its weights are released. Stop the service before launching an alternate runtime. Benchmark commands that use the existing HTTP endpoint share its one loaded model.

The tuner checks an arithmetic answer and times each placement serially. Its final JSON identifies the fastest measured placement for decode. Use `--prompt-tokens 1024` when comparing prefill and inspect the separate prompt and generation rates.

[Upstream expert cache](https://github.com/ggml-org/llama.cpp/blob/b12602/docs/moe-cache.md) · [Loop scale correction](https://github.com/ggml-org/llama.cpp/pull/32704) · [Mainline GGUF format](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Mainline_Support/README.md)


## Fit a 128k conversation

Inkling was trained for up to 1M tokens. Set `INKLING_CONTEXT` to allocate a slot inside that window. Its position mechanism uses a learned relative bias and log-scaled queries. There is no RoPE parameter to extend. The 40 local attention passes retain a 1,024-token sliding window. The eight global passes retain the entire prompt, each with its own keys and values. At 131,072 slots the FP16 global cache uses 4,096 MiB. With a 512-token microbatch, the live profile also allocates 240 MiB for local KV and 3.66 MiB for recurrent state. The earlier 32-token profile allocated 200 MiB for local KV. [Thinking Machines model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) · [Inkling attention implementation](https://github.com/ggml-org/llama.cpp/blob/b12602/src/models/inkling2.cpp)

The fresh-checkout default uses reference weights and FP16 keys and values. Set `INKLING_CACHE_TYPE=q8_0` to use Q8 for both. This reduces the estimated context buffers to 2,289 MiB and frees VRAM for experts. The CUDA build has a native Q8/Q8 Flash Attention decode kernel. Mixed Q8/FP16 falls back to an FP16 conversion. Q8 still stores every prompt position and can change the generated tokens. Pi's `/compact` instead summarizes old conversation turns. Use it when a shorter history is acceptable and prompt processing time matters.

```bash
./scripts/service stop
INKLING_CACHE_TYPE=q8_0 ./scripts/service start
```

For a smaller whole-device ceiling, `auto` measures expert-cache and staging allocations at the requested context. It tries to retain a microbatch of at least 64. When global KV banks move to RAM, hosted auto placement first tries copying the occupied prefix of each bank to a reusable CUDA buffer for attention. The memory estimator also considers CPU attention if staging cannot fit. Set `INKLING_STAGE_GLOBAL_KV=1` to require staging for host-backed global KV or `0` to use CPU attention for those banks. On a busier GPU it can also reduce the microbatch further or change dense-layer placement. Expert multiplies still execute on CUDA. Set `INKLING_BATCH`, `INKLING_UBATCH`, `INKLING_THREADS` and `INKLING_BATCH_THREADS` to tune logical batch size, maximum microbatch, decode threads and prompt threads independently. Set `INKLING_HOST_GLOBAL_KV` to a number from 0 to 8 to control how many global KV layers reside in RAM. Set `INKLING_HOST_OUTPUT=1` to put the vocabulary projection in RAM. Its CPU matmul runs on positions that need logits. The running service reports its placements, microbatch, expert-cache capacity in MiB and slots by weight type, reserve and VRAM peaks in `/api/ps`.

```bash
./scripts/probe-context --tokens 17000 --seed 959
./scripts/probe-context --tokens 131008 --seed 23591
```

Use `curl -s http://127.0.0.1:11434/api/ps` to inspect the live cache types and sampled whole-device and incremental VRAM peaks in MiB since startup. The probe counts the entire templated input, hides an eight-digit shelf label among other labels, and verifies the answer, token usage and streaming terminator. A digit-free assistant prefix starts the reply in Inkling’s text channel so that all 64 reserved output tokens are available for the answer. One cold near-capacity run occupies the single inference slot. For shorter controlled tests while the service is stopped, run `python3 inkling.py bench --context 131072 --prompt-tokens 1024 --tokens 64 --repeats 2 --batch 512 --ubatch 512 --threads 8 --batch-threads 16 --reserve-mib 1280`. For broader retrieval evaluation, the model publisher provides a UUID retrieval suite across depths and prompt lengths. [Inkling verification suite](https://github.com/thinking-machines-lab/inkling-verify)

### CUDA expert performance on the RTX 5070 Ti

The reference checkpoint uses a shared GPU expert cache and pinned host weights. Global attention evaluates groups of queries against the complete visible KV history. This limits the temporary space needed for the learned attention bias while letting expert projections process up to 512 prompt tokens together.

| Maximum microbatch | Cold prefill at 1,024 tokens (two runs, tokens/s) | Single-stream decode (median tokens/s) |
| ---: | ---: | ---: |
| 128 | 228, 227 | 17.3 |
| 256 | 286, 370 | 17.5 |
| 512 | 534, 546 | 17.5 |

These serial trials used a 131,072-token window, FP16 KV and 64 generated tokens per run. All attention and routed projections used CUDA. The 512-token trial peaked at 13.08 GB on the device, including 9.74 GB of growth above the desktop baseline. A separate FP16 trial with four global attention layers in RAM, a 256-token microbatch, and a larger GPU expert cache reached 358–364 prompt tokens/s and 19.4 decode tokens/s. Set `INKLING_HOST_GLOBAL_KV=4 INKLING_STAGE_GLOBAL_KV=0 INKLING_UBATCH=256` to try that placement. Both profiles use the reference weights. [Recorded trials](../benchmarks/rtx-5070-ti-gpu-experts.json)

In an earlier test with CPU attention enabled for host KV, a 10 GB device ceiling and a 131,072-token slot, auto placement moved all eight global attention passes to the CPU while keeping all routed expert multiplies on CUDA. A 256-token prompt reached 251 tokens/s, followed by 14.0 tokens/s for 24 generated tokens, at a 9.81 GB total peak. This trial measured short-input performance only.

### Earlier 128k measurements with CPU expert execution

The hosted reference model retrieved the correct label from **131,008 cold input tokens**. On the RTX 5070 Ti it processed 9,373 catalog rows in 33 min 5 s at 66.0 prompt tokens/s. It generated the six-token reply at 11.8 tokens/s and streamed a normal stop. The label was about 10% into the input. A separate 17,000-token probe passed in 4 min at 70.9 prompt tokens/s. Both used FP16 keys and values, 8 decode threads, 16 prefill threads and a 32-token microbatch. Peak whole-device VRAM across these probes was 12.03 GB, or 8.55 GB of growth from the desktop baseline. These figures measure one retrieval at each length. After the full run, the Ollama API checks and actual Pi and Codex file-tool round trips all passed on the same service with zero restarts. [Raw timings, memory and outputs](../benchmarks/rtx-5070-ti-128k.json)

The reference checkpoint on the RTX 5070 Ti gave these results with the original attention layout, a 2,304 MiB placement reserve, and one request at a time. These rows used two 48-token decode trials with a 12-token input. Peak memory includes the desktop.

| KV keys / values | Expert placement | Median decode (tokens/s) | Peak device (GB) |
| --- | --- | ---: | ---: |
| FP16 / FP16 | static | 16.8 | 11.89 |
| FP16 / FP16 | on-demand cache | 10.6 | 11.97 |
| Q8 / Q8 | static | 19.7 | 11.81 |
| Q8 / Q8 | on-demand cache | 17.5 | 11.97 |

The earlier graph added the relative-position bias to the attention mask in the gather buffer. At 128k this reduced estimated CUDA scratch from 816 to 624 MiB at microbatch 32 and from 1,633 to 1,249 MiB at microbatch 64. A fixed-placement CUDA comparison returned identical selected-token IDs and log probabilities on the first token after 1,408 input tokens and 16 tokens following a cached append.

A 1,024-token cold prefill with a 32-token microbatch and eight threads ran at 51.5 tokens/s with FP16 and 60.8 with Q8/Q8. Both passed arithmetic, JSON, Python and tool-use checks, and matched 24/24 next tokens at fixed prefixes. FP16 reproduced all eight 12-token reference continuations and Q8 reproduced six. Raising prompt threads to 16 gave 65.3 and 83.7 prompt tokens/s respectively in separate runs. Longer prompts become more expensive as global attention reads more history.

With a 10 GB whole-device ceiling and FP16, automatic placement put 25 of 49 offloadable layers on GPU and split the context buffers about equally between RAM and VRAM. With a 256-token input it processed 33.7 prompt tokens/s and generated 5.6 tokens/s. Sampled whole-device usage peaked at 7.16 GB with 3.96 GB of growth. [128k trial settings and results](../benchmarks/rtx-5070-ti-128k.json)

## Earlier 16k measurements

RTX 5070 Ti, Ryzen 9 9950X3D, WSL2, eight CPU threads, one active request, 16,384-token context. The table reports the median of two 96-token greedy decode trials after an arithmetic smoke test. Each row used the same reference checkpoint. Whole-device peaks include the Windows desktop. The two device-ceiling groups used prefill microbatches of 16 and 64, respectively.

| Total device ceiling | Expert placement | Decode tokens/s | Sampled total peak (GB) |
| --- | --- | ---: | ---: |
| 12 GB | hybrid | 20.4 | 11.74 |
| 12 GB | static | 24.2 | 10.50 |
| 12 GB | cache | 25.9 | 11.30 |
| 14 GB | hybrid | 24.1 | 12.86 |
| 14 GB | static | 30.6 | 13.60 |
| 14 GB | cache | 32.2 | 13.31 |

The optimized graph and the original b12602 graph with the upstream loop-scale fix produced **96/96 identical greedy token IDs** across eight short prompts spanning code, math, language and factual completions. Both loaded the same verified reference weights. The mean absolute difference in selected-token log probability was 0.0067 and the maximum was 0.1008. Run `./scripts/sample-quality` to export the same samples from a running endpoint. This check covers those prompts. Full evaluation-suite parity with the original model implementation remains unmeasured.

## What to run with a busy desktop

On the same RTX 5070 Ti with roughly 5.7 GiB already in use, we limited runtime growth to 7 GiB, retained the 14 GB device ceiling, and reserved 1,792 MiB during placement. Each checkpoint ran serially with the 16,384-token context. Speeds are median generated tokens per second over two 96-token trials. Each checkpoint passed all four local functional checks in the common cache configuration.

| Checkpoint | Cache | Static | Hybrid | Reference next tokens matched, out of 24 | Identical 12-token completions, out of 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 13.1 | 15.2 | 10.2 | baseline | baseline |
| Q4_K_XL | 20.4 | 18.4 | 16.2 | 24 | 6 |
| Q4_K_M | 20.5 | 18.7 | 16.9 | 24 | 7 |
| IQ3_XXS | 27.4 | 22.7 | 23.7 | 21 | 3 |

Pi v0.85.1 successfully executed a shell listing on both Q4_K_XL and the reference. The natural-language replies each had a small transcription error. Q4_K_XL omitted `shared-expert-cache.patch` and the reference misstated the repository directory in its heading. The underlying shell results were correct. These are single, sampled agent turns.

Two implementation comparisons used the same limits and cache placement. The original b12602 evaluation graph, rebuilt with the loop-scale correction and the same capped allocator and cache code, reached 12.8 tokens/s on the reference. The optimized graph reached 13.1. The Unsloth b12588 fork with the Q4_K_M target and a three-token DSpark draft reached 8.9 tokens/s. It accepted 48 of 137 proposed tokens in each timed trial. The optimized mainline Q4_K_M run reached 20.5. The fork comparison changes both the implementation and speculative decoding.

Choose the reference for the original checkpoint precision. Q4_K_XL is the balanced option for an interactive agent. Unsloth reports 98.0% top-token agreement with the reference across seven domains for Q4_K_XL, 97.9% for Q4_K_M, and 94.7% for IQ3_XXS. These percentages measure token decisions. Task success over a long coding session is a different measurement. [Publisher’s comparison](https://unsloth.ai/docs/models/inkling-2-mini#quantization-analysis) · [Local trial results](../benchmarks/rtx-5070-ti.json)

## Reproduce a checkpoint comparison

Stop the service before timing. The following Bash commands use the same context, microbatch, memory allowance, deterministic prompts, and two 96-token decode trials for every checkpoint. Each comparison also tests arithmetic, JSON output, Python understanding and a structured directory-listing tool call. The reference run supplies the prefix tokens for the other models. This lets a later probe compare identical input even if an earlier completion diverged.

```bash
mkdir -p .runtime/comparison
settings=(--context 16384 --ubatch 64 --device-gb 14 --vram-gib 7 --reserve-mib 1792 --tokens 96 --repeats 2)
python3 inkling.py compare --variant reference --mode cache "${settings[@]}" > .runtime/comparison/reference.json
for variant in q4xl q4 iq3; do
    python3 checkpoints.py "$variant"
    python3 inkling.py compare --variant "$variant" --mode cache "${settings[@]}" \
        --against .runtime/comparison/reference.json > ".runtime/comparison/$variant.json"
done
./scripts/service start
```

Compare placement independently with `inkling.py bench --variant VARIANT --mode MODE` using the same settings. Benchmark while the desktop has its normal workload. The amount of GPU memory available for experts makes a large difference on this machine.

## Source and model licenses

The llama.cpp license accompanies the upstream patches. Setup retrieves the published GGUF checkpoint separately. The model card specifies Apache 2.0 for the model. [llama.cpp MIT license](../LICENSE.llama.cpp) · [Inkling 2 Mini model card](https://thinkingmachines.ai/model-card/inkling-2-mini/)

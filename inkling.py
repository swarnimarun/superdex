#!/usr/bin/env python3
"""Single-GPU Inkling 2 Mini inference with measured VRAM budgets."""
import argparse
from collections import Counter
from contextlib import contextmanager
import errno
import fcntl
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from checkpoints import MODELS, RUNTIME
from expert_cache_status import ExpertCacheStatus, expert_cache_stats

BIN = RUNTIME / "llama-b12602"
CUDA = RUNTIME / "cudart-llama-b12602-bin-ubuntu-cuda-12.8-x64"
MODEL = MODELS["reference"].path
MODEL_ID = "inkling-2-mini:latest"
RUNTIME_STAMP = "gpu-experts-v8"
MAX_RUNTIME_MIB = 13_000_000_000 // (1024 ** 2)
HOST_RESERVE_MIB = 1280
GPU_ATTENTION_RESERVE_MIB = 768
GPU_ATTENTION_MIN_AVAILABLE_MIB = 9400
HOST_OUTPUT_RESERVE_MIB = 1536
HOST_OUTPUT_RULE = r"^output\.weight$=CPU"
GPU_OUTPUT_RULE = r"^output\.weight$=CUDA0"
# Small per-expert scales follow their layer's placement and are reused on CUDA.
EXPERT_WEIGHT_SUFFIX = r"\.ffn_(gate_up|up|down|gate)_(ch|)exps\.weight$=CPU"
HOST_EXPERT_WEIGHTS_RULE = r"^blk\.[0-9]+" + EXPERT_WEIGHT_SUFFIX


class PlacementDoesNotFit(RuntimeError):
    pass


@contextmanager
def single_model_worker(parent=Path("/tmp")):
    # All checkouts and GPU indices for this Linux user share the host weights budget.
    directory = parent / f"superdex-model-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"Expected a private model lock directory owned by this user: {directory}")
    with (directory / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A Superdex model worker is already running for this user") from error
        # The native child inherits this descriptor so it retains the reservation
        # even if its Python controller exits unexpectedly.
        yield lock.fileno()


def gpu_query(gpu):
    return ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.free,memory.used",
            "--format=csv,noheader,nounits"]


def configure_device_routes(env, directory):
    mode = env.get("INKLING_DEVICE_ROUTES", "0")
    if mode not in ("0", "1"):
        raise RuntimeError("INKLING_DEVICE_ROUTES must be 0 or 1")
    if mode == "0":
        return False, "off"
    plugin = directory / "libinkling-device-routes.so"
    if not plugin.is_file():
        env["INKLING_DEVICE_ROUTES"] = "0"
        return True, "unavailable"
    env["LD_PRELOAD"] = os.pathsep.join(filter(None, [str(plugin), env.get("LD_PRELOAD")]))
    return True, "pending"


def configure_decode_plugin(env, directory, variable, filename, selected):
    enabled = env.get(variable, "0")
    if enabled not in ("0", "1"):
        raise RuntimeError(f"{variable} must be 0 or 1")
    plugin = directory / filename
    if enabled == "0":
        return "off"
    if not plugin.is_file():
        env[variable] = "0"
        return "unavailable"
    env["LD_PRELOAD"] = os.pathsep.join(filter(None, [str(plugin), env.get("LD_PRELOAD")]))
    return selected


def configure_controller(env, directory):
    return configure_decode_plugin(env, directory, "INKLING_CONTROLLER_FUSION", "libinkling-controller.so", "sm120-qkv")


def configure_shortconv(env, directory):
    return configure_decode_plugin(env, directory, "INKLING_FUSED_SHORTCONV", "libinkling-shortconv.so", "sm120-four-tap")


def configure_compact_attention(env, directory):
    """Only the estimator and inference backend need the pinned ggml CUDA hook."""
    mode = env.get("INKLING_COMPACT_KERNEL", "")
    if not mode:
        return "external" if env.get("INKLING_COMPACT_FA") == "1" else "dense"
    if mode not in ("auto", "native", "hand", "off"):
        raise RuntimeError("INKLING_COMPACT_KERNEL must be auto, native, hand or off")
    if mode == "off" or env.get("INKLING_COMPACT_FA") == "0":
        env["INKLING_COMPACT_FA"] = "0"
        return "dense"
    backend = "native" if mode == "auto" else mode
    dispatcher = directory / "libinkling-compact-dispatch.so"
    kernel = directory / ("libinkling-native-fa.so" if backend == "native" else "libinkling-compact-fa.so")
    if not (dispatcher.is_file() and kernel.is_file()):
        if mode == "auto":
            env["INKLING_COMPACT_FA"] = "0"
            return "dense"
        raise RuntimeError(f"Build the requested attention kernel with ./scripts/build-compact-fa --{backend}")
    preload = [str(dispatcher)]
    if backend == "native":
        env["INKLING_NATIVE_FA_LIBRARY"] = str(kernel)
    else:
        env.pop("INKLING_NATIVE_FA_LIBRARY", None)
        preload.append(str(kernel))
    if env.get("LD_PRELOAD"):
        preload.append(env["LD_PRELOAD"])
    env["LD_PRELOAD"] = ":".join(preload)
    env["INKLING_COMPACT_FA"] = "1"
    return backend


def parse_gpu_info(gpu, output):
    try:
        uuid, free, used = (part.strip() for part in output.split(","))
        free, used = int(free), int(used)
        if not uuid or min(free, used) < 0:
            raise ValueError
        return uuid, free, used
    except ValueError as error:
        raise RuntimeError(f"No usable memory reading for GPU {gpu}: {output.strip()}") from error


def gpu_info(gpu):
    return parse_gpu_info(gpu, subprocess.check_output(gpu_query(gpu), text=True, timeout=10))


def watch_gpu(uuid, stop, observe, fail, timeout=10):
    """Check each complete sample immediately. Keep one NVML client for the session."""
    proc = None
    try:
        master, slave = os.openpty()
        # nvidia-smi can block-buffer stdout on a pipe. A terminal gives us each
        # CSV line promptly without relying on stdbuf or a second subprocess.
        with os.fdopen(master, "rb", buffering=0) as stream:
            with os.fdopen(slave, "wb", buffering=0) as terminal:
                proc = subprocess.Popen([*gpu_query(uuid), "--loop-ms=150"], stdin=subprocess.DEVNULL,
                                        stdout=terminal, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + timeout
            pending = b""
            while not stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"No complete memory reading for GPU {uuid} in {timeout:g}s")
                ready, _, _ = select.select([stream], [], [], min(0.15, remaining))
                if not ready or stop.is_set():
                    continue
                try:
                    chunk = stream.read(8192)
                except OSError as exc:
                    if exc.errno != errno.EIO:  # Linux PTYs report EIO after the writer exits.
                        raise
                    chunk = b""
                if not chunk:
                    detail = pending.decode(errors="replace").strip()
                    raise RuntimeError(f"nvidia-smi memory stream ended for GPU {uuid} (exit {proc.poll()}): {detail}")
                *lines, pending = (pending + chunk).split(b"\n")
                if len(pending) > 4096:
                    raise RuntimeError(f"Oversized nvidia-smi reading for GPU {uuid}")
                for line in lines:
                    if not line.strip():
                        continue
                    observed, free, used = parse_gpu_info(uuid, line.decode(errors="replace"))
                    if observed != uuid:
                        raise RuntimeError(f"nvidia-smi returned GPU {observed} while monitoring {uuid}")
                    observe(free, used)
                    deadline = time.monotonic() + timeout
    except Exception as exc:
        # Stop the backend before waiting for a malfunctioning telemetry process.
        fail(str(exc))
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def fit(args, env, available, *, wide_gpu_only=False):
    common = ["-m", str(args.model), "-c", str(args.context), "-b", str(args.batch),
              "-ub", str(args.ubatch), "-t", str(args.threads),
              "--no-op-offload", "--load-mode", args.load_mode, "-np", "1",
              "--moe-cache-cpu-share", "0",
              "--cache-type-k", args.cache_type_k, "--cache-type-v", args.cache_type_v]
    if args.batch_threads is not None:
        common.extend(["-tb", str(args.batch_threads)])

    @lru_cache(maxsize=None)
    def estimate(host_kv, ubatch, gpu_layers, *placement):
        opt = [*common, "-ngl", str(gpu_layers), *placement]
        output_rule = None
        if args.host_output == 1:
            output_rule = HOST_OUTPUT_RULE
        elif args.host_output == 0 and gpu_layers == 0:
            output_rule = GPU_OUTPUT_RULE
        if output_rule:
            if "-ot" in opt:
                opt[opt.index("-ot") + 1] += "," + output_rule
            else:
                opt.extend(["-ot", output_rule])
        opt[opt.index("-ub") + 1] = str(ubatch)
        result = subprocess.run([str(BIN / "llama-fit-params"), "--fit-print", "on", *opt],
                                env=dict(env, INKLING_MOE_CACHE_UBATCH_FLOOR=str(ubatch),
                                         INKLING_HOST_GLOBAL_KV=str(host_kv)),
                                text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError("CUDA estimator failed: " + result.stderr[-2000:])
        match = re.search(r"^CUDA0\s+(\d+)\s+(\d+)\s+(\d+)\s*$", result.stdout, re.M)
        if not match:
            raise RuntimeError("Missing CUDA estimate: " + result.stdout + result.stderr)
        return opt, tuple(map(int, match.groups()))

    # Inkling visits layers 6..17 three times per token. Keep these weights on GPU first.
    def host_layers(count):
        if count == 24:
            return ["-ot", HOST_EXPERT_WEIGHTS_RULE]
        order = [*range(23, 17, -1), *range(6), *range(6, 18)]
        if count == 0:
            return []
        numbers = "|".join(str(i) for i in order[:count])
        return ["-ot", rf"^blk\.({numbers})" + EXPERT_WEIGHT_SUFFIX]

    target = available - args.reserve_mib
    gpu_layers = 49
    ubatch = args.ubatch
    host_kv = args.host_global_kv or 0
    env["INKLING_HOST_GLOBAL_KV"] = str(host_kv)
    if args.mode == "auto":
        flags, parts = estimate(host_kv, ubatch, gpu_layers)
        if sum(parts) <= target:
            return flags, sum(parts)

    def cache_limit(host_kv, tile, placement):
        _, (weights, context, _) = estimate(host_kv, tile, gpu_layers, *placement)
        size = target - weights - context
        if args.cache_mib is not None:
            size = min(size, args.cache_mib)
        return size

    def cache_plan(host_kv, tile, placement, size):
        # An uncached host bank can need gigabytes of GPU staging. A cache can
        # eliminate that staging, so measure scratch with the cache in place.
        while size >= 256:
            flags, parts = estimate(host_kv, tile, gpu_layers, *placement, "--moe-cache-mib", str(size))
            projected = sum(parts)
            if projected <= target:
                return flags, projected
            size -= max(16, projected - target)
        return None

    if args.mode == "hybrid":
        # Test actual cached footprints. Uncached staging makes estimates for
        # host layers a poor proxy for the size of the finished hybrid graph.
        for count in range(12, 25):
            placement = host_layers(count)
            plan = cache_plan(host_kv, ubatch, placement, cache_limit(host_kv, ubatch, placement))
            if plan:
                return plan
    if args.mode in ("auto", "cache"):
        kv_placements = (host_kv,)
        if args.mode == "auto" and args.host_global_kv is None and not wide_gpu_only:
            kv_placements = (0, 2, 4, 6, 8)
        tiles = [ubatch]
        if args.mode == "auto":
            while tiles[-1] > 1:
                tiles.append(tiles[-1] // 2)
        # Keep attention on the GPU when it can sustain prompt batching. At low
        # budgets, try moving some KV to RAM before settling for token-sized batches.
        prefill_floor = min(64, ubatch)
        for group in ([t for t in tiles if t >= prefill_floor], [t for t in tiles if t < prefill_floor]):
            if not group or (wide_gpu_only and group[0] < prefill_floor):
                continue
            for host_kv in kv_placements:
                placements = [tuple(host_layers(24))]
                # Try a host vocabulary projection before transferring attention
                # to RAM. It only runs for positions that need output logits.
                if (args.mode == "auto" and args.host_output is None and host_kv == 0
                        and group[0] >= prefill_floor and args.reserve_mib >= HOST_OUTPUT_RESERVE_MIB):
                    placements.append(("-ot", HOST_EXPERT_WEIGHTS_RULE + "," + HOST_OUTPUT_RULE))
                # A single minimum-batch estimate bounds the cache budget for
                # this placement. Each candidate is verified at its actual batch.
                for placement in placements:
                    size = cache_limit(host_kv, tiles[-1], placement)
                    if size < 256:
                        continue
                    for tile in group:
                        plan = cache_plan(host_kv, tile, placement, size)
                        if plan:
                            env["INKLING_HOST_GLOBAL_KV"] = str(host_kv)
                            return plan
    if wide_gpu_only and args.mode != "static":
        return None
    if args.mode in ("cache", "hybrid"):
        raise PlacementDoesNotFit("Expert cache and GPU staging exceed the available budget")

    if args.mode == "auto":
        # Smaller devices may need some dense weights and KV in RAM. Expert
        # multiplies from those layers are still pinned to CUDA by the scheduler.
        low, high = 0, gpu_layers
        while low < high:
            middle = (low + high + 1) // 2
            if sum(estimate(host_kv, ubatch, middle, *host_layers(24))[1]) <= target:
                low = middle
            else:
                high = middle - 1
        gpu_layers = low

    # Static residency still streams host expert weights into CUDA staging buffers.
    for count in range(25):
        flags, parts = estimate(host_kv, ubatch, gpu_layers, *host_layers(count))
        if sum(parts) <= target:
            env["INKLING_HOST_GLOBAL_KV"] = str(host_kv)
            return flags, sum(parts)
    if wide_gpu_only:
        return None
    raise PlacementDoesNotFit("Dense weights and context exceed the available budget for minimum buffers")


def fit_with_reserve(args, env, available):
    # Each trial estimates the exact graph selected for the server, including the
    # reusable GPU buffers for staging full-attention KV from RAM.
    requested = args.stage_global_kv
    env["INKLING_STAGE_GLOBAL_KV"] = "1" if requested == "auto" else requested
    try:
        return fit_selected_attention(args, env, available)
    except PlacementDoesNotFit:
        if requested != "auto" or args.host_global_kv == 0:
            raise
    env["INKLING_STAGE_GLOBAL_KV"] = "0"
    return fit_selected_attention(args, env, available)


def fit_selected_attention(args, env, available):
    if args.reserve_mib is not None:
        flags, projected = fit(args, env, available)
        return flags, projected, args.reserve_mib

    trial = argparse.Namespace(**vars(args))
    if available >= GPU_ATTENTION_MIN_AVAILABLE_MIB and args.host_global_kv in (None, 0):
        # WSL's measured growth above the estimate was ~400 MiB with a GPU
        # vocabulary projection and ~1,242 MiB with a host vocabulary projection.
        trial.host_global_kv = 0
        for host_output, reserve in [(0, GPU_ATTENTION_RESERVE_MIB), (1, HOST_OUTPUT_RESERVE_MIB)]:
            if args.host_output is not None and host_output != args.host_output:
                continue
            if host_output and args.mode != "auto" and args.host_output is None:
                continue
            trial.host_output, trial.reserve_mib = host_output, reserve
            plan = fit(trial, env, available, wide_gpu_only=True)
            if plan:
                flags, projected = plan
                return flags, projected, reserve

    trial.reserve_mib = HOST_OUTPUT_RESERVE_MIB if args.host_output else HOST_RESERVE_MIB
    trial.host_output = args.host_output
    trial.host_global_kv = args.host_global_kv
    flags, projected = fit(trial, env, available)
    return flags, projected, trial.reserve_mib


def api(port, path, payload=None, timeout=300):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/{path}",
                                     data=json.dumps(payload).encode() if payload else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def host_attention_devices(startup_log):
    """Count actual attention backends for global KV banks allocated in RAM."""
    devices = {}
    for layer, target in re.findall(r"Inkling global KV: layer=(\d+) (?:device=CPU|buffer=\S+ attention=(\S+))", startup_log):
        devices[layer] = target or "CPU"
    if re.search(r"flash_attn\s+= disabled|Flash Attention not supported, set to disabled", startup_log):
        devices = dict.fromkeys(devices, "CPU")
    return dict(Counter(devices.values()))


def effective_load_mode(requested, startup_log):
    if requested != "dio":
        return requested
    direct = set(re.findall(r"file '([^\n]+?)' opened with direct I/O", startup_log))
    fallback = set(re.findall(r"file '([^\n]+?)' (?:with unbuffered I/O:[^\n]*\. Falling back to buffered I/O|falling back to buffered I/O)", startup_log))
    if fallback:
        return "mixed" if direct - fallback else "none"
    return "dio" if direct else "unknown"


def host_ram_stats(pid):
    """Read Linux process RSS and system availability in MiB, including pinned host buffers in RSS."""
    try:
        process = Path(f"/proc/{pid}/status").read_text()
        memory = Path("/proc/meminfo").read_text()
    except FileNotFoundError:
        return {}
    fields = {"VmRSS": "backend_rss_mib", "VmHWM": "backend_peak_rss_mib", "MemAvailable": "host_available_mib"}
    return {fields[key]: int(kib) // 1024 for key, kib in
            re.findall(r"^(VmRSS|VmHWM|MemAvailable):\s*(\d+) kB", process + memory, flags=re.M)}


def env_integer(env, name, default, minimum, maximum):
    try:
        value = int(env.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer from {minimum} to {maximum}") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def configure_prompt_lookup(env):
    mode = env.get("INKLING_PROMPT_LOOKUP", "none")
    if mode not in ("none", "ngram-simple", "ngram-map-k"):
        raise RuntimeError("INKLING_PROMPT_LOOKUP must be none, ngram-simple, or ngram-map-k")
    enabled = mode != "none"
    rows = env_integer(env, "INKLING_LOOKUP_ROLLBACK_ROWS", 4 if enabled else 0, int(enabled), 7)
    # b12602 ngram-simple declines every proposal shorter than the match length.
    max_match = rows if mode == "ngram-simple" else 1024
    match = env_integer(env, "INKLING_LOOKUP_MATCH_TOKENS", min(4, max_match), 1, max_match)
    delay = env_integer(env, "INKLING_DECODE_CACHE_MIN_TOKENS", 12, 1, 2**31 - 1)
    # Upstream appends speculative types supplied via CLI to those from the
    # environment, and can infer a draft implementation from a draft path.
    for key in ("LLAMA_ARG_SPEC_TYPE", "LLAMA_ARG_SPEC_DRAFT_MODEL", "LLAMA_ARG_SPEC_DRAFT_HF_REPO"):
        env.pop(key, None)
    env["INKLING_LOOKUP_ROLLBACK_ROWS"] = str(rows)
    env["INKLING_DECODE_CACHE_MIN_TOKENS"] = str(delay)
    flags = ["--spec-type", mode, f"--spec-{mode}-size-n", str(match),
             f"--spec-{mode}-size-m", str(rows)] if enabled else []
    return flags, {"prompt_lookup": mode, "lookup_match_tokens": match, "lookup_rollback_rows": rows,
                   "expert_cache_decode_min_tokens": delay}


def execute(args, worker_lock_fd):
    extra_cache_mib = env_integer(os.environ, "INKLING_DECODE_CACHE_EXTRA_MIB", 0, 0, 16384)
    cache_headroom_mib = env_integer(os.environ, "INKLING_DECODE_CACHE_HEADROOM_MIB", 0, 0, 16384)
    uuid, free, used = gpu_info(args.gpu)
    ceiling = int(args.device_gb * 1_000_000_000 // 1024**2)
    limit = min(MAX_RUNTIME_MIB, ceiling - used, int(args.vram_gib * 1024), free - 256)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, INKLING_GPU_EXPERTS="1",
               LD_LIBRARY_PATH=os.pathsep.join(
        str(p) for p in [BIN, CUDA, os.environ.get("LD_LIBRARY_PATH", "")] if p))
    plugin_dir = (RUNTIME / "compact-fa-native/current").resolve()
    if env.get("INKLING_BATCH_H2D") == "1":
        upload = plugin_dir / "libinkling-batch-h2d.so"
        if not upload.is_file():
            raise RuntimeError("Build the expert upload plug-in with ./scripts/build-compact-fa --native")
        env["LD_PRELOAD"] = os.pathsep.join(filter(None, [str(upload), env.get("LD_PRELOAD")]))
    graph_routes = env.get("INKLING_GRAPH_ROUTES", "0")
    if graph_routes not in ("0", "1"):
        raise RuntimeError("INKLING_GRAPH_ROUTES must be 0 or 1")
    if graph_routes == "1":
        env.setdefault("INKLING_DEVICE_ROUTES", "1")  # an explicit ownership opt-out wins
    routes_requested, routes_state = configure_device_routes(env, plugin_dir)
    controller = configure_controller(env, plugin_dir)
    shortconv = configure_shortconv(env, plugin_dir)
    compact_kernel = configure_compact_attention(env, plugin_dir)
    gqa_min_rows = env_integer(env, "INKLING_GQA_MIN_K", 0, 0, 2**31 - 1)
    env["INKLING_GQA_MIN_K"] = str(gqa_min_rows)
    env["INKLING_DECODE_CACHE_EXTRA_MIB"] = str(extra_cache_mib)
    env["INKLING_DECODE_CACHE_HEADROOM_MIB"] = str(cache_headroom_mib)
    lookup_flags, lookup_stats = configure_prompt_lookup(env)
    elastic_mode = env.get("INKLING_ELASTIC_KV", "0")
    if elastic_mode not in ("0", "auto", "1"):
        raise RuntimeError("INKLING_ELASTIC_KV must be 0, auto, or 1")
    elastic_requested = elastic_mode != "0"
    env["INKLING_ELASTIC_KV"] = elastic_mode
    env["INKLING_ELASTIC_KV_CHUNK_MIB"] = str(env_integer(env, "INKLING_ELASTIC_KV_CHUNK_MIB", 0, 0, 32))
    for _ in range(3):
        flags, projected, reserve = fit_with_reserve(args, env, limit)
        # Desktop workloads may grow while the metadata-only estimator runs.
        _, baseline, used = gpu_info(uuid)
        limit = min(limit, baseline - 256, ceiling - used)
        if args.reserve_mib is None and reserve == GPU_ATTENTION_RESERVE_MIB and limit < GPU_ATTENTION_MIN_AVAILABLE_MIB:
            continue
        if projected <= limit - reserve:
            break
    else:
        raise RuntimeError("GPU memory availability kept changing during placement")
    real_ubatch = int(flags[flags.index("-ub") + 1])
    if lookup_flags and lookup_stats["lookup_rollback_rows"] >= real_ubatch:
        raise RuntimeError(f"Prompt lookup needs a microbatch larger than its rollback row count. "
                           f"Placement gave {real_ubatch}, INKLING_LOOKUP_ROLLBACK_ROWS={lookup_stats['lookup_rollback_rows']}")
    env["INKLING_MOE_CACHE_UBATCH_FLOOR"] = str(real_ubatch)
    host_kv = int(env["INKLING_HOST_GLOBAL_KV"])
    gpu_layers = int(flags[flags.index("-ngl") + 1])
    overrides = flags[flags.index("-ot") + 1].split(",") if "-ot" in flags else []
    host_output = HOST_OUTPUT_RULE in overrides or (gpu_layers == 0 and GPU_OUTPUT_RULE not in overrides)
    peak = 0
    device_peak = used
    errors = []
    stop = threading.Event()
    with tempfile.NamedTemporaryFile(prefix="inkling-", suffix=".log", dir=RUNTIME, delete=False) as log:
        print(f"budget={limit} MiB estimated_buffers={projected} MiB reserve={reserve} MiB log={log.name}\n"
              + " ".join(flags), file=sys.stderr, flush=True)
        # GGML device buffers are admitted before allocation. Leave a separate
        # allowance for the driver, library workspaces and allocator rounding.
        env["INKLING_BUFFER_LIMIT_BYTES"] = str((limit - min(512, reserve)) * 1024**2)
        proc = subprocess.Popen([str(BIN / "llama-server"), *flags, *lookup_flags, "--fit", "off",
                                 "--cache-ram", str(args.cache_ram_mib), "--ctx-checkpoints", str(args.ctx_checkpoints),
                                 "--cache-prompt",
                                 "--alias", MODEL_ID, "--jinja", "--temp", "1", "--top-k", "0",
                                 "--top-p", "1", "--min-p", "0", "--host", "127.0.0.1", "--port", str(args.port), "-lv", "4",
                                 "--reasoning-budget", str(args.reasoning_budget)],
                                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                pass_fds=(worker_lock_fd,))

        def observe(remaining, used_now):
            nonlocal peak, device_peak
            device_peak = max(device_peak, used_now)
            peak = max(peak, baseline - remaining)
            if used_now > ceiling:
                raise RuntimeError(f"GPU reports {used_now} MiB used, exceeding the {ceiling} MiB device limit")
            if peak > limit:
                raise RuntimeError(f"Observed GPU memory rose {peak} MiB above baseline (limit {limit} MiB)")

        def fail_monitor(message):
            errors.append(message)
            proc.kill()
            print(f"GPU watchdog stopped the backend: {message}", file=sys.stderr, flush=True)

        watcher = threading.Thread(target=watch_gpu, args=(uuid, stop, observe, fail_monitor), daemon=True)
        watcher.start()
        try:
            deadline = time.monotonic() + 240
            while proc.poll() is None:
                # Confirm this process bound the port before sending it requests.
                if b"listening on http://" in Path(log.name).read_bytes():
                    if api(args.port, "health", timeout=5).get("status") == "ok":
                        break
                if time.monotonic() > deadline:
                    raise RuntimeError("Server startup timed out")
                time.sleep(0.2)
            if proc.poll() is not None:
                detail = Path(log.name).read_text(errors="replace")[-1600:]
                raise RuntimeError(f"Server exited with code {proc.returncode}\n{detail}")
            startup_bytes = Path(log.name).read_bytes()
            startup_log = startup_bytes.decode(errors="replace")
            base_cache_mib = int(flags[flags.index("--moe-cache-mib") + 1]) if "--moe-cache-mib" in flags else 0
            track_phases = bool(extra_cache_mib and base_cache_mib and args.action in ("host", "bench", "compare"))
            track_log = track_phases or routes_state == "pending" or elastic_requested or (args.action == "host" and (compact_kernel != "dense" or controller == "sm120-qkv" or shortconv == "sm120-four-tap"))
            cache_status = ExpertCacheStatus(startup_log, Path(log.name) if track_log else None,
                                            len(startup_bytes), dynamic=track_phases)
            if track_log:
                threading.Thread(target=cache_status.follow, args=(stop,), daemon=True).start()
            placement_stats = {
                "load_mode": effective_load_mode(args.load_mode, startup_log), "requested_load_mode": args.load_mode,
                "fast_relative_positions": env.get("INKLING_FAST_REL_POS") == "1" or env.get("INKLING_PACK_GLOBAL_MASK") == "1",
                "packed_global_mask_requested": env.get("INKLING_PACK_GLOBAL_MASK") == "1",
                "compact_attention_requested": compact_kernel,
                "grouped_decode_min_kv_rows": gqa_min_rows,
                "elastic_global_kv_requested": elastic_requested, "elastic_global_kv_mode": elastic_mode,
                "elastic_global_kv_active": False,
                "controller_fusion_requested": controller,
                "shortconv_requested": shortconv,
                "device_routes_requested": routes_requested, "device_routes_state": routes_state,
                "device_routes_active": False,
                "device_routes_graph_requested": graph_routes == "1", "device_routes_graph_active": False,
                "cache_ram_mib": args.cache_ram_mib, "ctx_checkpoints": args.ctx_checkpoints,
                "stage_global_kv": env["INKLING_STAGE_GLOBAL_KV"] == "1",
                "host_global_attention": host_attention_devices(startup_log),
                "expert_cache_mib": base_cache_mib, "expert_cache_decode_extra_limit_mib": extra_cache_mib,
                "expert_cache_decode_headroom_limit_mib": cache_headroom_mib,
                **lookup_stats,
            }
            if args.action == "host":
                from local_api import Handler
                from http.server import ThreadingHTTPServer
                with ThreadingHTTPServer(("127.0.0.1", args.api_port), Handler) as gateway:
                    gateway.backend_port = args.port
                    gateway.context = args.context
                    gateway.checkpoint = MODELS[args.variant]
                    gateway.runtime_stats = lambda: {
                        **host_ram_stats(proc.pid),
                        "expert_compute": "CUDA0",
                        "host_global_kv": host_kv, "host_output": host_output,
                        **placement_stats, **cache_status.snapshot(),
                        "cache_type_k": args.cache_type_k, "cache_type_v": args.cache_type_v,
                        "batch": args.batch, "ubatch": real_ubatch,
                        "gpu_layers": gpu_layers,
                        "device_limit_mib": ceiling, "device_peak_mib": device_peak,
                        "growth_limit_mib": limit, "growth_peak_mib": peak, "reserve_mib": reserve,
                    }
                    gateway.timeout = 0.5
                    print(f"Ready at http://127.0.0.1:{args.api_port}", flush=True)
                    while proc.poll() is None:
                        gateway.handle_request()
                    raise RuntimeError("Inference backend stopped")
            elif args.action == "serve":
                print(f"Ready at http://127.0.0.1:{args.port}/v1", flush=True)
                if proc.wait():
                    raise RuntimeError("Server stopped unexpectedly")
            elif args.action in ("bench", "compare"):
                check = api(args.port, "v1/chat/completions", {
                    "messages": [{"role": "user", "content": "What is 7 times 8? Reply with the number."}],
                    "temperature": 0, "max_tokens": 256,
                })
                answer = check["choices"][0]["message"].get("content") or ""
                if not re.search(r"\b56\b", answer):
                    raise RuntimeError(f"Model failed the arithmetic smoke test: {answer!r}")
                # Fixed-length decode, identical prompt and no KV reuse across repetitions.
                payload = {"prompt": "Here is a detailed explanation of how rain forms. First,",
                           "n_predict": args.tokens, "temperature": 0, "ignore_eos": True,
                           "cache_prompt": False, "seed": 42}
                if args.prompt_tokens:
                    seed = api(args.port, "tokenize", {"content": (
                        "Rain gauges measure daily precipitation. Weather stations report the air "
                        "temperature and pressure each hour. Observers compare those readings.\n"),
                        "add_special": False})["tokens"]
                    payload["prompt"] = (seed * math.ceil(args.prompt_tokens / len(seed)))[:args.prompt_tokens]
                measurements = [api(args.port, "completion", payload, timeout=10800)["timings"]
                                for _ in range(args.repeats)]
                if any(m["predicted_n"] != args.tokens for m in measurements):
                    raise RuntimeError("Incomplete benchmark decode")
                result = {
                    **host_ram_stats(proc.pid),
                    "mode": args.mode, "cpu_share": args.cpu_share,
                    "context": args.context, "batch": args.batch, "ubatch": real_ubatch,
                    "cache_type_k": args.cache_type_k, "host_global_kv": host_kv, "host_output": host_output,
                    **placement_stats, **cache_status.snapshot(),
                    "expert_compute": "CUDA0",
                    "cache_type_v": args.cache_type_v, "threads": args.threads,
                    "batch_threads": args.batch_threads or args.threads,
                    "placement_flags": flags[flags.index("-ngl"):],
                    "budget_mib": limit, "reserve_mib": reserve, "sampled_peak_mib": peak, "device_peak_mib": device_peak, "smoke_answer": answer,
                    "decode_tps_median": statistics.median(m["predicted_per_second"] for m in measurements),
                    "trials": measurements,
                }
                if args.action == "compare":
                    from comparison import compare
                    reference = json.loads(args.against.read_text())["quality"] if args.against else None
                    result["quality"] = compare(lambda path, data: api(args.port, path, data), reference)
                    result.update(model=args.model.name, context=args.context, sampled_peak_mib=peak,
                                  device_peak_mib=device_peak)
                print(json.dumps(result), flush=True)
                return result
            else:
                payload = {"messages": [{"role": "user", "content": args.prompt}],
                           "temperature": 0, "seed": 42, "max_tokens": args.tokens}
                data = api(args.port, "v1/chat/completions", payload)
                print(json.dumps({"message": data["choices"][0]["message"], "finish_reason":
                      data["choices"][0]["finish_reason"], "timings": data["timings"],
                      "sampled_peak_mib": peak, "budget_mib": limit}, ensure_ascii=False), flush=True)
        finally:
            stop.set()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            watcher.join()
            print(f"sampled_peak={peak} MiB device_peak={device_peak} MiB", file=sys.stderr)
            if errors:
                raise RuntimeError(errors[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "serve", "host", "bench", "compare", "tune"])
    p.add_argument("--against", type=Path, help="Reference compare JSON for teacher-forced token probes")
    p.add_argument("--prompt", default="What is 7 times 8? Reply with the number.")
    p.add_argument("--vram-gib", type=float, default=MAX_RUNTIME_MIB / 1024)
    p.add_argument("--device-gb", type=float, default=14, help="Total device ceiling in decimal GB, including other processes (max 15)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--mode", choices=["auto", "static", "cache", "hybrid"], default="auto")
    p.add_argument("--cache-type-k", choices=["f16", "q8_0", "q4_0"], default="f16",
                   help="Attention key-cache format")
    p.add_argument("--cache-type-v", choices=["f16", "q8_0", "q4_0"], default="f16",
                   help="Attention value-cache format")
    p.add_argument("--cache-mib", type=int)
    p.add_argument("--cache-ram-mib", type=int, default=os.getenv("INKLING_CACHE_RAM_MIB", "0"),
                   help="RAM limit in MiB for saved prompts across conversations (default 0)")
    p.add_argument("--ctx-checkpoints", type=int, default=os.getenv("INKLING_CTX_CHECKPOINTS", "0"),
                   help="Maximum intermediate context snapshots per slot (default 0)")
    host_kv_env = os.getenv("INKLING_HOST_GLOBAL_KV", "auto")
    p.add_argument("--host-global-kv", type=int, choices=range(9),
                   default=None if host_kv_env == "auto" else host_kv_env,
                   help="Full-attention KV layers stored in RAM (0..8); auto mode starts on GPU")
    p.add_argument("--stage-global-kv", choices=("auto", "0", "1"), default=os.getenv("INKLING_STAGE_GLOBAL_KV"),
                   help="Attention on host-backed KV: CPU (0), layer device (1), or prefer staging within budget (auto)")
    p.add_argument("--host-output", type=int, choices=(0, 1),
                   help="Keep the vocabulary projection on GPU (0) or in host RAM (1); auto mode tries GPU first")
    p.add_argument("--load-mode", choices=["dio", "mmap", "none"], default=os.getenv("INKLING_LOAD_MODE"),
                   help="Read weights with direct I/O (dio), buffered I/O (none), or a file mapping (mmap)")
    p.add_argument("--cpu-share", type=float, choices=[0], default=0, help="Routed expert execution is GPU-only")
    p.add_argument("--reserve-mib", type=lambda s: None if s == "auto" else int(s), default=1024,
                   help="Placement headroom in MiB, or auto for adaptive hosted headroom (CLI default 1024)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-threads", type=int, help="CPU threads for prompt processing (default: same as --threads)")
    p.add_argument("--batch", type=int, default=64, help="Maximum logical prompt batch sent to the backend")
    p.add_argument("--ubatch", type=int, default=8, help="Maximum tokens evaluated per physical microbatch")
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prompt-tokens", type=int, help="Benchmark prefill with exactly this many input token IDs")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--reasoning-budget", type=int, default=-1)
    p.add_argument("--api-port", type=int, default=11434)
    p.add_argument("--port", type=int, default=18567)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path)
    source.add_argument("--variant", choices=MODELS, default="reference")
    a = p.parse_args()
    if a.load_mode is None:
        a.load_mode = "dio" if a.action == "host" else "none"
    if a.load_mode not in ("dio", "mmap", "none"):
        p.error("INKLING_LOAD_MODE must be dio, mmap or none")
    if not (0 <= a.cache_ram_mib <= 2**31 - 1 and 0 <= a.ctx_checkpoints <= 2**31 - 1):
        p.error("The saved prompt cache size and context checkpoint count must be integers from 0 to 2147483647")
    if a.stage_global_kv is None:
        a.stage_global_kv = "auto" if a.action == "host" else "0"
    if a.stage_global_kv not in ("0", "1", "auto") or a.host_global_kv not in (None, *range(9)):
        p.error("INKLING_STAGE_GLOBAL_KV must be 0, 1 or auto and INKLING_HOST_GLOBAL_KV must be 0..8 or auto")
    if not (0 < a.device_gb <= 15 and math.isfinite(a.vram_gib) and a.vram_gib > 0 and
            (a.reserve_mib is None or a.reserve_mib >= 0) and
            0 <= a.cpu_share <= 1 and 0 < a.ubatch <= a.batch <= 2**31 - 1 and a.threads > 0 and
            (a.batch_threads is None or a.batch_threads > 0) and
            a.context > 0 and a.tokens > 0 and a.repeats > 0 and 0 < a.port <= 65535 and
            (a.prompt_tokens is None or 0 < a.prompt_tokens <= a.context - a.tokens) and
            0 < a.api_port <= 65535 and (a.action != "host" or a.api_port != a.port) and
            (a.cache_mib is None or a.cache_mib >= 256)):
        p.error("Invalid budget, size, share, port or thread count")
    a.model = a.model or MODELS[a.variant].path
    if a.action == "host" and a.model.resolve() != MODELS[a.variant].path:
        p.error("Select the hosted checkpoint with --variant")
    RUNTIME.mkdir(exist_ok=True)
    setup_lock_path = RUNTIME / ".setup.lock"
    gpu_lock_path = RUNTIME / f".gpu-{a.gpu}.lock"
    with single_model_worker() as worker_lock_fd, setup_lock_path.open("a") as lock, gpu_lock_path.open("a") as gpu_lock:
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("This checkout already has a server or benchmark on the selected GPU")
        fcntl.flock(lock, fcntl.LOCK_SH)
        stamp = BIN / ".inkling-ready"
        if not a.model.is_file() or not stamp.is_file() or stamp.read_text().strip() != RUNTIME_STAMP:
            p.error("Complete ./setup.sh before launching")
        if a.action == "tune":
            results = []
            for mode, share in [("static", 0), ("hybrid", 0), ("cache", 0)]:
                trial = argparse.Namespace(**vars(a))
                trial.mode, trial.cpu_share, trial.action = mode, share, "bench"
                try:
                    results.append(execute(trial, worker_lock_fd))
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    print(f"Skipping {mode} cpu_share={share}: {exc}", file=sys.stderr)
            if not results:
                raise RuntimeError("No configuration passed the benchmark")
            best = max(results, key=lambda r: r["decode_tps_median"])
            print(json.dumps({"fastest_measured": best}), flush=True)
        else:
            execute(a, worker_lock_fd)


if __name__ == "__main__":
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        sys.exit(str(error))

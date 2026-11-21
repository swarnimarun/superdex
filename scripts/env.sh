# Source with Bash to connect local tools to Superdex.
inkling_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z ${INKLING_PROFILE:-} && -f "$inkling_repo/.runtime/host-profile" ]]; then
    IFS= read -r INKLING_PROFILE < "$inkling_repo/.runtime/host-profile" || true
fi
if [[ ${INKLING_PROFILE:-standard} == fast ]]; then
    export INKLING_FAST_REL_POS="${INKLING_FAST_REL_POS:-1}"
    export INKLING_PACK_GLOBAL_MASK="${INKLING_PACK_GLOBAL_MASK:-1}"
    export INKLING_VRAM_GB="${INKLING_VRAM_GB:-15}"
    export INKLING_BATCH="${INKLING_BATCH:-1536}"
    export INKLING_UBATCH="${INKLING_UBATCH:-$INKLING_BATCH}"
    export INKLING_COMPACT_KERNEL="${INKLING_COMPACT_KERNEL:-auto}"
    export INKLING_NATIVE_FA_MATCH_TILES="${INKLING_NATIVE_FA_MATCH_TILES:-1}"
    export INKLING_MODE="${INKLING_MODE:-cache}"
    if [[ -f "$inkling_repo/.runtime/compact-fa-native/current/libinkling-native-fa.so" && $INKLING_COMPACT_KERNEL != off && ${INKLING_COMPACT_FA:-1} != 0 ]]; then
        export INKLING_ELASTIC_KV="${INKLING_ELASTIC_KV:-auto}"
        export INKLING_ELASTIC_KV_CHUNK_MIB="${INKLING_ELASTIC_KV_CHUNK_MIB:-8}"
        export INKLING_CONTROLLER_FUSION="${INKLING_CONTROLLER_FUSION:-1}"
        export INKLING_DEVICE_ROUTES="${INKLING_DEVICE_ROUTES:-1}"
        export INKLING_GRAPH_ROUTES="${INKLING_GRAPH_ROUTES:-1}"
        export GGML_CUDA_GRAPH_CHUNK_SIZE="${GGML_CUDA_GRAPH_CHUNK_SIZE:-1024}"
        export INKLING_GQA_MIN_K="${INKLING_GQA_MIN_K:-65536}"
        export INKLING_COMPACT_DECODE="${INKLING_COMPACT_DECODE:-1}"
        export INKLING_RESERVE_MIB="${INKLING_RESERVE_MIB:-1024}"
        export INKLING_DECODE_CACHE_EXTRA_MIB="${INKLING_DECODE_CACHE_EXTRA_MIB:-6144}"
        export INKLING_DECODE_CACHE_HEADROOM_MIB="${INKLING_DECODE_CACHE_HEADROOM_MIB:-768}"
    else
        export INKLING_CACHE_MIB="${INKLING_CACHE_MIB:-1037}"
        export INKLING_RESERVE_MIB="${INKLING_RESERVE_MIB:-896}"
        export INKLING_DECODE_CACHE_EXTRA_MIB="${INKLING_DECODE_CACHE_EXTRA_MIB:-3500}"
    fi
    export INKLING_HOST_GLOBAL_KV="${INKLING_HOST_GLOBAL_KV:-0}"
    export INKLING_HOST_OUTPUT="${INKLING_HOST_OUTPUT:-0}"
fi
unset inkling_repo
export INKLING_PORT="${INKLING_PORT:-11434}"
export INKLING_BACKEND_PORT="${INKLING_BACKEND_PORT:-18567}"
export INKLING_CONTEXT="${INKLING_CONTEXT:-131072}"
export INKLING_VRAM_GB="${INKLING_VRAM_GB:-14}"
export INKLING_GPU="${INKLING_GPU:-0}"
export INKLING_RESERVE_MIB="${INKLING_RESERVE_MIB:-auto}"
export INKLING_CACHE_TYPE_K="${INKLING_CACHE_TYPE:-${INKLING_CACHE_TYPE_K:-f16}}"
export INKLING_CACHE_TYPE_V="${INKLING_CACHE_TYPE:-${INKLING_CACHE_TYPE_V:-f16}}"
export INKLING_CACHE_RAM_MIB="${INKLING_CACHE_RAM_MIB:-0}"
export INKLING_CTX_CHECKPOINTS="${INKLING_CTX_CHECKPOINTS:-0}"
export INKLING_HOST_GLOBAL_KV="${INKLING_HOST_GLOBAL_KV:-auto}"
export INKLING_STAGE_GLOBAL_KV="${INKLING_STAGE_GLOBAL_KV:-auto}"
export INKLING_HOST_OUTPUT="${INKLING_HOST_OUTPUT:-auto}"
export INKLING_THREADS="${INKLING_THREADS:-8}"
if [[ -z ${INKLING_BATCH_THREADS:-} ]]; then
    INKLING_BATCH_THREADS=$(nproc)
    if ((INKLING_BATCH_THREADS > 16)); then INKLING_BATCH_THREADS=16; fi
fi
export INKLING_BATCH_THREADS
export INKLING_CPU_SHARE="${INKLING_CPU_SHARE:-0}"
export INKLING_MODE="${INKLING_MODE:-auto}"
export INKLING_LOAD_MODE="${INKLING_LOAD_MODE:-dio}"
if [[ -z ${INKLING_VARIANT:-} ]]; then
    INKLING_VARIANT=reference
    if [[ ${INKLING_PROFILE:-standard} == fast ]]; then INKLING_VARIANT=q4; fi
    variant_file="$(dirname "${BASH_SOURCE[0]}")/../.runtime/host-variant"
    if [[ -f $variant_file ]] && IFS= read -r saved_variant < "$variant_file"; then
        case "$saved_variant" in reference|q4xl|q4|iq3) INKLING_VARIANT=$saved_variant ;; esac
    fi
    unset saved_variant variant_file
fi
export INKLING_VARIANT
inkling_batch_default=512
if [[ $INKLING_VARIANT == q4xl && $INKLING_VRAM_GB =~ ^15([.]0+)?$ ]]; then
    inkling_batch_default=1024
fi
export INKLING_BATCH="${INKLING_BATCH:-$inkling_batch_default}"
export INKLING_UBATCH="${INKLING_UBATCH:-$INKLING_BATCH}"
unset inkling_batch_default
if [[ $INKLING_VARIANT == q4xl ]]; then
    inkling_mask_default=1
else
    inkling_mask_default=0
fi
export INKLING_FAST_REL_POS="${INKLING_FAST_REL_POS:-$inkling_mask_default}"
export INKLING_PACK_GLOBAL_MASK="${INKLING_PACK_GLOBAL_MASK:-$inkling_mask_default}"
unset inkling_mask_default
export INKLING_MODEL='inkling-2-mini:latest'
export OLLAMA_HOST="http://127.0.0.1:$INKLING_PORT"
export INKLING_BASE_URL="$OLLAMA_HOST/v1"

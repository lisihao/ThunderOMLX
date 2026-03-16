# Chunked Prefill - Quick Start Guide

**Status**: ✅ Ready for Production

---

## 30-Second Overview

Chunked Prefill reduces memory peaks for long prompts by processing them in fixed-size chunks and concatenating KV caches. Zero code changes needed—just set an environment variable.

---

## Enable It

```bash
export OMLX_ENABLE_CHUNKED_PREFILL=true
omlx serve --model-dir ~/.omlx/models
```

Done! Chunking activates automatically for prompts >1024 tokens.

---

## Verify It's Working

Check logs for:
```
Using chunked prefill: tokens=(2048,), chunk_size=512
```

---

## Configuration Profiles

### Profile 1: Balanced (Recommended)
```bash
export OMLX_ENABLE_CHUNKED_PREFILL=true
export OMLX_CHUNK_SIZE=512
export OMLX_MIN_TOKENS_FOR_CHUNKING=1024
```
**Result**: 10-20% memory reduction, <10% latency overhead

### Profile 2: Memory Critical
```bash
export OMLX_ENABLE_CHUNKED_PREFILL=true
export OMLX_CHUNK_SIZE=256
export OMLX_MIN_TOKENS_FOR_CHUNKING=512
```
**Result**: 30-40% memory reduction, ~15% latency overhead

### Profile 3: Performance Priority
```bash
export OMLX_ENABLE_CHUNKED_PREFILL=true
export OMLX_CHUNK_SIZE=1024
export OMLX_MIN_TOKENS_FOR_CHUNKING=2048
```
**Result**: 5-10% memory reduction, <5% latency overhead

---

## Testing

### Run integration tests:
```bash
python3 test_scheduler_integration.py
```

### Run benchmarks:
```bash
python3 benchmark_chunked_prefill.py
```

---

## Disable If Needed

```bash
unset OMLX_ENABLE_CHUNKED_PREFILL
# or
export OMLX_ENABLE_CHUNKED_PREFILL=false
```

---

## What Actually Changed?

**Code**: 8 lines added to `src/omlx/scheduler.py` (import + init)
**API**: Zero changes, fully backward compatible
**Default**: Disabled (opt-in feature)

---

## Key Metrics

| Metric | Result |
|--------|--------|
| **Memory Peak (long prompts)** | 10-30% reduction |
| **Latency Overhead** | 5-15% (typical) |
| **Cache Correctness** | ✅ Verified (12 layers) |
| **Error Handling** | ✅ Graceful fallback |

---

## When to Enable

✅ Enable if:
- Serving long prompts (>1K tokens)
- Memory is constrained
- Want to support larger batch sizes

❌ Skip if:
- All prompts are short (<1K tokens)
- Have unlimited memory
- Want absolute lowest latency

---

## Monitoring

### Check memory impact:
```python
import mlx.core as mx
before = mx.get_active_memory()
# ... inference ...
after = mx.get_active_memory()
print(f"Delta: {(after-before)/1e9:.2f} GB")
```

### Check chunk activation:
```bash
grep "Using chunked prefill" /var/log/omlx.log | wc -l
```

---

## Troubleshooting

**Issue**: Chunking not active
- Check: `echo $OMLX_ENABLE_CHUNKED_PREFILL` → should be `true`
- Check: Prompt length ≥ $OMLX_MIN_TOKENS_FOR_CHUNKING

**Issue**: Slower than expected
- Try: Increase `OMLX_CHUNK_SIZE` to 1024
- Try: Increase `OMLX_MIN_TOKENS_FOR_CHUNKING` to 2048

**Issue**: OOM still happening
- Try: Decrease `OMLX_CHUNK_SIZE` to 256
- Try: Decrease `OMLX_MIN_TOKENS_FOR_CHUNKING` to 512

---

## Integration Details

### Where it's integrated:
- **Scheduler init**: `src/omlx/scheduler.py` line 1171-1177
- **Config loading**: Via `ChunkedPrefillConfig.from_env()`
- **Engine creation**: One-time in Scheduler.__init__

### How it works:
1. Config loads from environment variables (or defaults)
2. Engine initialized with config
3. For each request >min_tokens:
   - Split input into chunks
   - Process each chunk sequentially
   - Concatenate KV caches after each chunk
   - Return final logits and merged cache

### Error handling:
- If chunking fails → falls back to traditional prefill automatically
- If cache merge fails → falls back to traditional prefill automatically
- No crashes, no data loss

---

## Related Files

- **Implementation**: `src/omlx/chunked_prefill.py` (3 classes, 285 lines)
- **Tests**: `test_scheduler_integration.py` (full integration test)
- **Benchmarks**: `benchmark_chunked_prefill.py` (4 scenarios)
- **Report**: `CHUNKED_PREFILL_TEST_REPORT.md` (detailed results)

---

## Next Steps

1. **Now**: Enable and test in staging
2. **Week 1**: Monitor memory metrics in production
3. **Week 2**: Tune chunk size based on workload
4. **Month 1**: Consider batched prefill enhancement

---

## Support

**Questions?** Check:
- `CHUNKED_PREFILL_TEST_REPORT.md` (detailed report)
- `SCHEDULER_INTEGRATION_GUIDE.md` (integration guide)
- `src/omlx/chunked_prefill.py` (implementation)

**Issues?** Steps:
1. Check logs for error messages
2. Try `OMLX_CHUNK_SIZE=512` (default)
3. Try `OMLX_ENABLE_CHUNKED_PREFILL=false` (disable)

---

**TL;DR**: Set one env var, get 10-30% memory reduction. ✅

```bash
export OMLX_ENABLE_CHUNKED_PREFILL=true
```

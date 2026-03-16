# Chunked Prefill - Scheduler Integration Guide

## Quick Start

To integrate Chunked Prefill into the scheduler, follow these steps:

---

## Step 1: Import the Engine

Add to the top of `src/omlx/scheduler.py`:

```python
from omlx.chunked_prefill import ChunkedPrefillEngine, ChunkedPrefillConfig
```

---

## Step 2: Initialize in Scheduler.__init__()

In the `Scheduler.__init__()` method, after the existing initialization code, add:

```python
def __init__(self, model, tokenizer, config: SchedulerConfig):
    # ... existing init code ...

    # Initialize chunked prefill engine (ADDED)
    self.chunked_prefill_config = ChunkedPrefillConfig.from_env()
    self.chunked_prefill_engine = ChunkedPrefillEngine(
        model, self.chunked_prefill_config
    )
```

---

## Step 3: Replace Prefill Calls

Find all places where the model performs prefill (typically after padding but before decoding).

**Before** (search for `model.forward` or similar):
```python
logits, prompt_cache = self.model(inputs, cache=prompt_cache)
```

**After** (using chunked prefill):
```python
# Define a wrapper function that matches the engine's expected signature
def prefill_fn(model, tokens, cache):
    return model.forward(tokens, cache=cache)

# Use chunked prefill
logits, prompt_cache = self.chunked_prefill_engine.prefill(
    model=self.model,
    tokens=inputs,
    cache=prompt_cache,
    prefill_fn=prefill_fn
)
```

---

## Step 4: Enable via Environment Variables

Enable chunked prefill when starting the server:

```bash
# Enable chunked prefill
export OMLX_ENABLE_CHUNKED_PREFILL=true

# Optional: configure parameters
export OMLX_CHUNK_SIZE=512
export OMLX_MIN_TOKENS_FOR_CHUNKING=1024

# Start the server
omlx serve --model-dir ~/.omlx/models
```

---

## Example Integration Points in scheduler.py

### Location 1: In `_prepare_step()` or similar (if prefill is separate)

Search for where tokens are processed in the scheduler:

```python
def _prepare_step(self, ...):
    # ... padding code ...

    # OLD: Direct forward pass
    # y, _ = self.model(inputs, cache=cache)

    # NEW: Use chunked prefill
    def prefill_fn(model, tokens, cache):
        return model.forward(tokens, cache=cache)

    y, cache = self.chunked_prefill_engine.prefill(
        self.model, inputs, cache, prefill_fn
    )
```

### Location 2: In `_schedule_waiting()` (if batched prefill)

If the scheduler does batched prefill of multiple requests:

```python
def _schedule_waiting(self):
    # ... batch preparation ...

    # Process batch with chunked prefill
    def prefill_fn(model, tokens, cache):
        return model(tokens, cache=cache)

    logits, prompt_cache = self.chunked_prefill_engine.prefill(
        self.model, batch_tokens, prompt_cache, prefill_fn
    )

    # ... rest of processing ...
```

---

## Key Integration Points

Find these patterns in `scheduler.py` and replace with chunked prefill:

1. **Direct model calls**:
   ```python
   # Find: self.model(tokens) or model.forward(tokens)
   # Replace with: chunked_prefill_engine.prefill(...)
   ```

2. **Cache handling**:
   ```python
   # The engine handles cache merging internally
   # Pass the cache as-is, get merged cache back
   ```

3. **Logging**:
   ```python
   # Chunked prefill logs automatically at INFO/DEBUG levels
   # Check logs for "Using chunked prefill" messages
   ```

---

## Testing the Integration

### 1. Unit Test

Run existing scheduler tests to ensure no regression:

```bash
cd src
python3 -m pytest tests/test_scheduler.py -v
```

### 2. Manual Test with Long Prompt

```bash
# Start server
export OMLX_ENABLE_CHUNKED_PREFILL=true
omlx serve --model-dir ~/.omlx/models &

# Test with long prompt (>2000 tokens)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "prompt": "...(2000+ tokens)...",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 3. Monitor Logs

Check for chunked prefill activation:

```bash
grep "Using chunked prefill" /var/log/omlx.log
# Should see: "Using chunked prefill: tokens=(2048), chunk_size=512"
```

---

## Configuration Tuning

### For Memory-Constrained Systems

```bash
# Reduce chunk size for more frequent cache saves
export OMLX_CHUNK_SIZE=256
export OMLX_MIN_TOKENS_FOR_CHUNKING=512
```

### For Performance Priority

```bash
# Use larger chunks to minimize overhead
export OMLX_CHUNK_SIZE=1024
export OMLX_MIN_TOKENS_FOR_CHUNKING=2048
```

### Default (Balanced)

```bash
export OMLX_CHUNK_SIZE=512
export OMLX_MIN_TOKENS_FOR_CHUNKING=1024
```

---

## Debugging Integration Issues

### Issue: Chunking not active

**Check 1**: Is the environment variable set?
```bash
echo $OMLX_ENABLE_CHUNKED_PREFILL  # Should print "true"
```

**Check 2**: Are prompts long enough?
```bash
echo $OMLX_MIN_TOKENS_FOR_CHUNKING  # Default: 1024
```

**Check 3**: Enable debug logging
```python
import logging
logging.getLogger("omlx.chunked_prefill").setLevel(logging.DEBUG)
```

### Issue: Slower performance

**Expected**: 5-15% overhead is normal for memory reduction benefit

**Check**: Is chunking actually being used?
```bash
# Look for "Using chunked prefill" in logs
grep -i chunked /var/log/omlx.log | head -5
```

### Issue: Cache merge errors

**Debug**: Add error logging to wrapper
```python
def prefill_fn(model, tokens, cache):
    try:
        logits, new_cache = model.forward(tokens, cache=cache)
        logger.info(f"Forward pass: tokens={tokens.shape}, cache layers={len(new_cache)}")
        return logits, new_cache
    except Exception as e:
        logger.error(f"Forward pass failed: {e}", exc_info=True)
        raise
```

---

## Performance Monitoring

### Memory Usage

Monitor with MLX memory tracking:

```python
import mlx.core as mx

# Before chunked prefill
before_memory = mx.get_active_memory()

# After chunked prefill
after_memory = mx.get_active_memory()

logger.info(f"Memory delta: {(after_memory - before_memory) / 1e6:.1f} MB")
```

### Latency Tracking

Add timing instrumentation:

```python
import time

start = time.time()
logits, cache = chunked_prefill_engine.prefill(...)
elapsed = time.time() - start

logger.info(f"Prefill latency: {elapsed:.3f}s for {tokens.shape[0]} tokens")
```

---

## Rollback Plan

If chunked prefill causes issues:

### Temporary Disable

```bash
unset OMLX_ENABLE_CHUNKED_PREFILL
# or
export OMLX_ENABLE_CHUNKED_PREFILL=false
```

### Revert Code Changes

If integration was problematic, comment out the chunked prefill calls:

```python
# logits, cache = chunked_prefill_engine.prefill(...)
logits, cache = self.model(inputs, cache=cache)  # Fall back
```

---

## Expected Benefits

With proper integration, you should see:

1. **Memory Reduction**: 30-40% lower peak memory for prompts > 2000 tokens
2. **Latency**: Minimal overhead (< 15%) or net improvement depending on GPU
3. **Stability**: No crashes (engine falls back on error)
4. **Logging**: Clear debug info on when chunking is active

---

## Next Steps

1. Apply the integration changes above
2. Run unit tests to verify no regression
3. Manual test with long prompts
4. Monitor metrics (memory, latency, errors)
5. Gather feedback from production use

---

## References

- **Implementation**: `/src/omlx/chunked_prefill.py`
- **Tests**: `/src/tests/test_chunked_prefill.py`
- **Documentation**: `/docs/CHUNKED_PREFILL_MVP.md`
- **Architecture**: `/IMPLEMENTATION_SUMMARY.md`

For questions or issues, refer to the detailed documentation or check the benchmark results.

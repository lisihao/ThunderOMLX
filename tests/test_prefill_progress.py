# SPDX-License-Identifier: Apache-2.0
"""Tests for Prefill Progress Streaming (P2.5 Phase 3)."""

import sys
import os
import queue
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_request_output_prefill_progress():
    """Test RequestOutput has prefill_progress field."""
    from omlx.request import RequestOutput

    # Default is None
    output = RequestOutput(request_id="test-1")
    assert output.prefill_progress is None

    # Can set progress
    output = RequestOutput(
        request_id="test-1",
        finished=False,
        prefill_progress={"processed_tokens": 4096, "total_tokens": 32768},
    )
    assert output.prefill_progress["processed_tokens"] == 4096
    assert output.prefill_progress["total_tokens"] == 32768
    assert not output.finished


def test_progress_queue_thread_safety():
    """Test queue-based progress passing is thread-safe."""
    progress_queue = queue.Queue(maxsize=500)

    events_written = []

    def writer():
        for i in range(100):
            try:
                progress_queue.put_nowait({"processed": i * 100, "total": 10000})
                events_written.append(i)
            except queue.Full:
                pass

    # Write from multiple threads
    threads = [threading.Thread(target=writer) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Drain queue
    events_read = []
    while True:
        try:
            events_read.append(progress_queue.get_nowait())
        except queue.Empty:
            break

    assert len(events_read) == len(events_written)
    assert len(events_read) > 0


def test_progress_queue_overflow():
    """Test queue overflow behavior (non-blocking, drops events)."""
    small_queue = queue.Queue(maxsize=5)

    dropped = 0
    for i in range(10):
        try:
            small_queue.put_nowait(i)
        except queue.Full:
            dropped += 1

    assert dropped == 5
    assert small_queue.qsize() == 5


def test_on_prefill_progress_creates_events():
    """Test _on_prefill_progress creates RequestOutput events."""
    from omlx.request import RequestOutput

    progress_queue = queue.Queue(maxsize=500)

    # Simulate uid_to_request_id mapping
    uid_to_request_id = {1: "req-abc", 2: "req-def"}

    # Simulate the callback logic
    def on_prefill_progress(progress_list):
        for uid, processed, total in progress_list:
            rid = uid_to_request_id.get(uid)
            if not rid:
                continue
            event = RequestOutput(
                request_id=rid,
                finished=False,
                prefill_progress={"processed_tokens": processed, "total_tokens": total},
            )
            try:
                progress_queue.put_nowait(event)
            except queue.Full:
                pass

    # Call with progress data
    on_prefill_progress([(1, 2048, 32768), (2, 2048, 16384)])

    events = []
    while True:
        try:
            events.append(progress_queue.get_nowait())
        except queue.Empty:
            break

    assert len(events) == 2
    assert events[0].request_id == "req-abc"
    assert events[0].prefill_progress["processed_tokens"] == 2048
    assert events[0].prefill_progress["total_tokens"] == 32768
    assert events[1].request_id == "req-def"
    assert events[1].prefill_progress["processed_tokens"] == 2048


def test_unknown_uid_ignored():
    """Test that unknown UIDs are silently ignored."""
    from omlx.request import RequestOutput

    progress_queue = queue.Queue(maxsize=500)
    uid_to_request_id = {1: "req-abc"}

    def on_prefill_progress(progress_list):
        for uid, processed, total in progress_list:
            rid = uid_to_request_id.get(uid)
            if not rid:
                continue
            event = RequestOutput(
                request_id=rid,
                finished=False,
                prefill_progress={"processed_tokens": processed, "total_tokens": total},
            )
            try:
                progress_queue.put_nowait(event)
            except queue.Full:
                pass

    # UID 999 is not in mapping
    on_prefill_progress([(1, 1024, 8192), (999, 512, 4096)])

    events = []
    while True:
        try:
            events.append(progress_queue.get_nowait())
        except queue.Empty:
            break

    # Only UID 1 should produce an event
    assert len(events) == 1
    assert events[0].request_id == "req-abc"


def test_generation_output_prefill_progress():
    """Test GenerationOutput has prefill_progress field."""
    from omlx.engine.base import GenerationOutput

    # Default is None
    output = GenerationOutput(text="hello")
    assert output.prefill_progress is None

    # Can set progress
    output = GenerationOutput(
        text="",
        new_text="",
        finished=False,
        prefill_progress={"processed_tokens": 8192, "total_tokens": 65536},
    )
    assert output.prefill_progress["processed_tokens"] == 8192
    assert output.prefill_progress["total_tokens"] == 65536


if __name__ == "__main__":
    tests = [
        test_request_output_prefill_progress,
        test_progress_queue_thread_safety,
        test_progress_queue_overflow,
        test_on_prefill_progress_creates_events,
        test_unknown_uid_ignored,
        test_generation_output_prefill_progress,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} tests failed")

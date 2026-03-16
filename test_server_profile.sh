#!/bin/bash
set -e

cd /Users/lisihao/ThunderOMLX

echo "🚀 Starting ThunderOMLX server with profiling..."
./venv/bin/python run_server_with_profile.py > /tmp/omlx-profile-server.log 2>&1 &
SERVER_PID=$!

echo "   Server PID: $SERVER_PID"
echo "   Log: /tmp/omlx-profile-server.log"
echo ""

# Wait for server to start
echo "⏳ Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "✅ Server ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        tail -50 /tmp/omlx-profile-server.log
        exit 1
    fi
    sleep 2
done

echo ""
echo "📊 Running 5 test requests to generate profile data..."
./venv/bin/python3 << 'PYTHON'
import asyncio
from openai import AsyncOpenAI
import time

async def run_tests():
    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
    prompt = "The quick brown fox jumps over the lazy dog. " * 200
    
    for i in range(5):
        print(f"  Request {i+1}/5...")
        start = time.time()
        
        response = await client.chat.completions.create(
            model="qwen3.5-35b-mlx",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
            stream=True
        )
        
        tokens = 0
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                tokens += 1
        
        elapsed = time.time() - start
        print(f"     {tokens} tokens in {elapsed:.2f}s ({tokens/elapsed:.1f} tok/s)")
    
    print("✅ All requests completed")

asyncio.run(run_tests())
PYTHON

echo ""
echo "🛑 Stopping server to generate profile..."
kill -TERM $SERVER_PID

# Wait for server to save profile
sleep 5

echo ""
if [ -f "server_process_profile.txt" ]; then
    echo "✅ Profile generated successfully!"
    echo ""
    echo "Top 10 functions by cumulative time:"
    head -40 server_process_profile.txt | tail -20
    echo ""
    echo "Full profile: server_process_profile.txt"
    echo "Self-time profile: server_process_profile_tottime.txt"
else
    echo "❌ Profile not generated"
    echo "Server log:"
    tail -100 /tmp/omlx-profile-server.log
fi

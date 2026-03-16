#!/bin/bash
set -e

SERVER_PID=$(ps aux | grep "omlx serve" | grep -v grep | awk '{print $2}')

if [ -z "$SERVER_PID" ]; then
    echo "❌ ThunderOMLX server not running"
    exit 1
fi

echo "🔍 Found ThunderOMLX server PID: $SERVER_PID"
echo "📊 Starting py-spy profiling for 30 seconds..."
echo ""

# Profile server process for 30 seconds
sudo ./venv/bin/py-spy record \
    --pid $SERVER_PID \
    --duration 30 \
    --format speedscope \
    --output server_process_profile.speedscope.json \
    --subprocesses &

PYSPY_PID=$!
echo "py-spy PID: $PYSPY_PID"
echo ""

# Wait 2 seconds for py-spy to attach
sleep 2

# Send test requests during profiling
echo "🚀 Sending API requests to trigger server activity..."
python3 << 'PYTHON'
import asyncio
from openai import AsyncOpenAI

async def send_requests():
    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
    prompt = "The quick brown fox jumps over the lazy dog. " * 200
    
    for i in range(5):
        print(f"  Request {i+1}/5...")
        response = await client.chat.completions.create(
            model="qwen3.5-35b-mlx",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
            stream=True
        )
        
        async for chunk in response:
            pass  # Consume stream
        
        await asyncio.sleep(1)
    
    print("✅ All requests completed")

asyncio.run(send_requests())
PYTHON

echo ""
echo "⏳ Waiting for py-spy to finish..."
wait $PYSPY_PID

echo ""
echo "✅ Profile complete: server_process_profile.speedscope.json"
echo "   View at: https://www.speedscope.app/"

import asyncio
import socket
from openai import AsyncOpenAI

async def main():
    print("Testing connection to LM Studio...")
    
    # 1. Debug DNS
    try:
        host = "host.docker.internal"
        ip = socket.gethostbyname(host)
        print(f"DNS Resolved {host} -> {ip}")
    except Exception as e:
        print(f"DNS Resolution failed: {e}")
        return

    # 2. Configure Client with explicit timeout
    print(f"Connecting to http://{ip}:2222/v1 ...")
    client = AsyncOpenAI(
        base_url=f"http://{ip}:2222/v1",
        api_key="lmstudio",
        timeout=120.0
    )
    
    try:
        print(f"Sending request to {client.base_url} for model 'qwen/qwen3-vl-8b'...")
        response = await client.chat.completions.create(
            model="qwen/qwen3-vl-8b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("Response received:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

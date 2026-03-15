"""Runnable demos for anymodel Python SDK.

Usage:
    python examples/basic.py              # Run all examples
    python examples/basic.py stream       # Run a specific example
    python examples/basic.py tools
    python examples/basic.py batch
    python examples/basic.py fallback
    python examples/basic.py models
    python examples/basic.py stats

Requires API keys set as environment variables:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_API_KEY=AIza...
"""

from __future__ import annotations

import asyncio
import sys

from anymodel import AnyModel


async def demo_completion():
    """Basic chat completion."""
    print("\n=== Chat Completion ===\n")
    client = AnyModel()

    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "What is the capital of France? Reply in one sentence."}],
    )
    print(response["choices"][0]["message"]["content"])
    print(f"\nTokens: {response['usage']}")


async def demo_stream():
    """Streaming chat completion."""
    print("\n=== Streaming ===\n")
    client = AnyModel()

    stream = await client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Write a haiku about programming."}],
        stream=True,
    )

    async for chunk in stream:
        content = chunk["choices"][0].get("delta", {}).get("content", "")
        print(content, end="", flush=True)
    print()


async def demo_tools():
    """Tool calling."""
    print("\n=== Tool Calling ===\n")
    client = AnyModel()

    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "What's the weather in New York City?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }],
        tool_choice="auto",
    )

    message = response["choices"][0]["message"]
    if message.get("tool_calls"):
        for call in message["tool_calls"]:
            print(f"Tool: {call['function']['name']}")
            print(f"Args: {call['function']['arguments']}")
    else:
        print(message.get("content", ""))


async def demo_fallback():
    """Fallback routing across providers."""
    print("\n=== Fallback Routing ===\n")
    client = AnyModel()

    response = await client.chat.completions.create(
        model="",
        models=[
            "anthropic/claude-sonnet-4-6",
            "openai/gpt-4o",
            "google/gemini-2.5-pro",
        ],
        route="fallback",
        messages=[{"role": "user", "content": "Say hello and tell me which model you are."}],
    )

    print(response["choices"][0]["message"]["content"])
    print(f"\nModel used: {response['model']}")


async def demo_batch():
    """Batch processing."""
    print("\n=== Batch Processing ===\n")
    client = AnyModel()

    print("Submitting batch...")
    results = await client.batches.create_and_poll(
        {
            "model": "anthropic/claude-haiku-4-5",
            "requests": [
                {"custom_id": "req-1", "messages": [{"role": "user", "content": "What is AI? One sentence."}]},
                {"custom_id": "req-2", "messages": [{"role": "user", "content": "What is ML? One sentence."}]},
                {"custom_id": "req-3", "messages": [{"role": "user", "content": "What is NLP? One sentence."}]},
            ],
        },
        on_progress=lambda b: print(f"  Progress: {b['completed']}/{b['total']}"),
    )

    print(f"\nBatch status: {results['status']}")
    for r in results["results"]:
        content = r["response"]["choices"][0]["message"]["content"] if r["response"] else r["error"]
        print(f"  {r['custom_id']}: {content}")


async def demo_models():
    """List available models."""
    print("\n=== Available Models ===\n")
    client = AnyModel()

    models = await client.models.list()
    for m in models[:10]:
        print(f"  {m['id']}")
    if len(models) > 10:
        print(f"  ... and {len(models) - 10} more")


async def demo_stats():
    """Generation stats."""
    print("\n=== Generation Stats ===\n")
    client = AnyModel()

    response = await client.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hi"}],
    )

    gen_id = response["id"]
    stats = client.generation.get(gen_id)
    if stats:
        print(f"  ID: {stats['id']}")
        print(f"  Model: {stats['model']}")
        print(f"  Latency: {stats['latency']:.2f}s")
        print(f"  Prompt tokens: {stats['tokens_prompt']}")
        print(f"  Completion tokens: {stats['tokens_completion']}")
    else:
        print("  No stats recorded")


DEMOS = {
    "completion": demo_completion,
    "stream": demo_stream,
    "tools": demo_tools,
    "fallback": demo_fallback,
    "batch": demo_batch,
    "models": demo_models,
    "stats": demo_stats,
}


async def main():
    args = sys.argv[1:]

    if args:
        for name in args:
            if name in DEMOS:
                await DEMOS[name]()
            else:
                print(f"Unknown demo: {name}")
                print(f"Available: {', '.join(DEMOS.keys())}")
                sys.exit(1)
    else:
        for demo_fn in DEMOS.values():
            await demo_fn()


if __name__ == "__main__":
    asyncio.run(main())

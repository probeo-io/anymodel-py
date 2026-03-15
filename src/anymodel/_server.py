"""HTTP server mode using Starlette (optional dependency)."""

from __future__ import annotations

import json
from typing import Any

from anymodel._client import AnyModel


def create_anymodel_app(config: dict[str, Any] | None = None) -> Any:
    """Create a Starlette ASGI app for the anymodel API.

    Requires the [server] extra: pip install anymodel[server]
    """
    try:
        from starlette.applications import Starlette
        from starlette.middleware.cors import CORSMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse
        from starlette.routing import Route
    except ImportError:
        raise ImportError(
            "Server mode requires starlette. Install with: pip install anymodel[server]"
        )

    client = AnyModel(config)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()

        if body.get("stream"):
            stream = await client.chat.completions.create(**body)

            async def sse_generator():
                async for chunk in stream:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        response = await client.chat.completions.create(**body)
        return JSONResponse(response)

    async def list_models(request: Request) -> JSONResponse:
        provider = request.query_params.get("provider")
        models = await client.models.list(provider=provider)
        return JSONResponse({"object": "list", "data": models})

    async def get_generation(request: Request) -> JSONResponse:
        gen_id = request.path_params["id"]
        stats = client.generation.get(gen_id)
        if not stats:
            return JSONResponse(
                {"error": {"code": 404, "message": f"Generation {gen_id} not found", "metadata": {}}},
                status_code=404,
            )
        return JSONResponse(stats)

    async def create_batch(request: Request) -> JSONResponse:
        body = await request.json()
        batch = await client.batches.create(body)
        return JSONResponse(batch, status_code=201)

    async def list_batches(request: Request) -> JSONResponse:
        batches = await client.batches.list()
        return JSONResponse({"object": "list", "data": batches})

    async def get_batch(request: Request) -> JSONResponse:
        batch_id = request.path_params["id"]
        batch = await client.batches.get(batch_id)
        if not batch:
            return JSONResponse(
                {"error": {"code": 404, "message": f"Batch {batch_id} not found", "metadata": {}}},
                status_code=404,
            )
        return JSONResponse(batch)

    async def get_batch_results(request: Request) -> JSONResponse:
        batch_id = request.path_params["id"]
        results = await client.batches.results(batch_id)
        return JSONResponse(results)

    async def cancel_batch(request: Request) -> JSONResponse:
        batch_id = request.path_params["id"]
        batch = await client.batches.cancel(batch_id)
        return JSONResponse(batch)

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/api/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/api/v1/models", list_models, methods=["GET"]),
        Route("/api/v1/generation/{id}", get_generation, methods=["GET"]),
        Route("/api/v1/batches", create_batch, methods=["POST"]),
        Route("/api/v1/batches", list_batches, methods=["GET"]),
        Route("/api/v1/batches/{id}", get_batch, methods=["GET"]),
        Route("/api/v1/batches/{id}/results", get_batch_results, methods=["GET"]),
        Route("/api/v1/batches/{id}/cancel", cancel_batch, methods=["POST"]),
    ]

    app = Starlette(routes=routes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

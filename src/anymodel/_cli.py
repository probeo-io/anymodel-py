"""CLI entry point for anymodel."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="anymodel",
        description="OpenRouter-compatible LLM router",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP server")
    serve_parser.add_argument("--port", type=int, default=4141, help="Port to listen on (default: 4141)")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")

    args = parser.parse_args()

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            print("Server mode requires uvicorn. Install with: pip install anymodel[server]", file=sys.stderr)
            sys.exit(1)

        from anymodel._server import create_anymodel_app

        app = create_anymodel_app()
        print(f"anymodel server running at http://{args.host}:{args.port}")
        print(f"API base: http://{args.host}:{args.port}/api/v1")
        print()
        print("Endpoints:")
        print("  POST /api/v1/chat/completions")
        print("  GET  /api/v1/models")
        print("  GET  /api/v1/generation/:id")
        print("  POST /api/v1/batches")
        print("  GET  /api/v1/batches")
        print("  GET  /api/v1/batches/:id")
        print("  GET  /api/v1/batches/:id/results")
        print("  POST /api/v1/batches/:id/cancel")
        print("  GET  /health")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

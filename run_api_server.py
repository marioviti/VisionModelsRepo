#!/usr/bin/env python3
"""
Quick start script for Vision Models API Server.

Usage:
    python run_api_server.py [--port PORT] [--host HOST] [--reload]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Start Vision Models API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed.")
        print("Please install API dependencies:")
        print("  pip install -r requirements.api.txt")
        sys.exit(1)

    print("=" * 70)
    print("Vision Models API Server")
    print("=" * 70)
    print(f"Server will start at: http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Reload enabled: {args.reload}")
    print(f"Workers: {args.workers}")
    print("=" * 70)
    print()

    # Import to check if package is installed
    try:
        from vision_model_repo.api.server import app
    except ImportError:
        print("ERROR: vision_model_repo package is not installed.")
        print("Please install the package:")
        print("  pip install -e .")
        sys.exit(1)

    # Start server
    uvicorn.run(
        "vision_model_repo.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1  # Can't use workers with reload
    )


if __name__ == "__main__":
    main()

"""Launch the ROPA review web app.

Usage::

    python -m pii_detection.ropa.review --db sqlite:///ropa.db --port 8000
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    """Parse CLI options and start the uvicorn server."""
    parser = argparse.ArgumentParser(description="ROPA review web app.")
    parser.add_argument("--db", default=None, help="database URL (e.g. sqlite:///ropa.db)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.db:
        os.environ["ROPA_DB_URL"] = args.db

    import uvicorn

    uvicorn.run("pii_detection.ropa.review.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

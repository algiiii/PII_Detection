"""Launch the DPO web app (compliance dashboard + mounted ROPA review).

Usage::

    python -m pii_detection.web --host 0.0.0.0 --port 8000

Inside the container the host must be ``0.0.0.0`` so the published port is
reachable; locally it defaults to ``127.0.0.1``. Database URLs are read from the
environment (``PII_DB_URL``, ``ROPA_DB_URL``).
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse CLI options and start the uvicorn server."""
    parser = argparse.ArgumentParser(description="PII compliance DPO web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("pii_detection.web.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

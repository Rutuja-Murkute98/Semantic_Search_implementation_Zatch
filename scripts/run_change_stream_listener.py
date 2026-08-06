"""
run_change_stream_listener.py — entrypoint for the real-time embedding
sync worker (see app/change_stream_listener.py for the actual logic).

Deployed as a persistent background worker (see the `worker` service in
render.yaml) -- unlike the cron backfill, this process runs continuously
and never exits on its own; it's what makes new/edited products, sellers,
and bits searchable by meaning within seconds instead of waiting for the
next scheduled backfill.

Run standalone:
    python scripts/run_change_stream_listener.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.change_stream_listener import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()

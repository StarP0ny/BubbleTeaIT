"""Compatibility entry point for running the worker from a checkout."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/bubbleteait/_workers/service_worker.py"), run_name="__main__")

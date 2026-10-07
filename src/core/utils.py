import time
from contextlib import contextmanager

@contextmanager
def timer(label: str):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"⏱️  [{label}] Latency: {duration:.4f}s")
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("portfolio")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


def log_event(event: str, trace_id: str | None = None, **fields: Any) -> None:
    payload = {
        "ts": time.time(),
        "event": event,
        "trace_id": trace_id or str(uuid.uuid4()),
        **fields,
    }
    logger.info(json.dumps(payload, default=str))


@contextmanager
def trace(operation: str, trace_id: str | None = None, **fields: Any) -> Iterator[dict[str, Any]]:
    tid = trace_id or str(uuid.uuid4())
    start = time.perf_counter()
    log_event(f"{operation}_start", trace_id=tid, **fields)
    ctx: dict[str, Any] = {"trace_id": tid}
    try:
        yield ctx
    except Exception:
        raise
    finally:
        latency = (time.perf_counter() - start) * 1000
        log_event(f"{operation}_end", trace_id=tid, latency_ms=latency, **fields)
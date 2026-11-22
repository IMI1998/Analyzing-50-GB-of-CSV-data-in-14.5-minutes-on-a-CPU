# server.py
# Titan Control Plane Server v5.9.0-Enterprise — SJF+Aging + Zero-I/O Scandir + O(1) Financial KPIs
import os
import time
import random
import shutil
import asyncio
import threading
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

from telemetry_math import ManifestLedger, ZeroOverheadProfiler
from engine import (
    TitanEngine,
    BASE_DIR,
    INBOX_DIR,
    SAVED_INBOX_DIR,
    FAILED_INBOX_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TitanServer")

profiler = ZeroOverheadProfiler()
ledger = ManifestLedger(os.path.join(BASE_DIR, "manifest.db"))
engine = TitanEngine(profiler, ledger, base_dir=BASE_DIR)

_worker_stop = threading.Event()
_worker_thread: Optional[threading.Thread] = None

_query_cache_lock = threading.Lock()
_query_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

# Background cached dashboard state (prevents WebSocket event-loop & disk I/O storms)
_state_cache_lock = threading.Lock()
_state_cache_ts: float = 0.0
_state_cache_payload: Dict[str, Any] = {}
_force_dir_rescan: bool = True

# In-memory HTML cache to avoid disk reads during heavy 50GB file ingestion
_html_cache_mtime: float = -1.0
_html_cache_content: str = ""

# SJF Aging tracking (prevents starvation of large 50GB files)
_first_seen_lock = threading.Lock()                          # ← اصلاح (1)
_first_seen_ts: Dict[str, float] = {}
_AGING_MB_PER_SEC: float = 5.0  # Every 1s waiting reduces effective sort size by 5MB
_MAX_GENERATE_MB: int = 65536   # Safety cap: 64 GB max per API call


# ==============================================================================
# Helper: Low-Overhead Directory Scanner (os.scandir avoids extra stat syscalls)
# ==============================================================================
def _is_file_stable_and_ready(path: str) -> bool:
    """Ensures external files being copied into inbox/ are completely written before ingestion."""
    try:
        st = os.stat(path)
        if st.st_size <= 0:
            return False
        if (time.time() - st.st_mtime) < 0.15:
            return False
        with open(path, "rb"):
            pass
        return True
    except OSError:
        return False


def _scan_csv_dir(folder: str, apply_aging: bool = False) -> List[Dict[str, Any]]:
    if not os.path.exists(folder):
        return []
    now = time.time()
    out: List[Dict[str, Any]] = []
    seen_names = set()

    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                name = entry.name
                if not name.endswith(".csv") or name.startswith(".") or name.endswith(".tmp") or name.endswith(".claiming"):
                    continue
                try:
                    st = entry.stat()
                    sz_bytes = int(st.st_size)
                    sz_mb = sz_bytes / (1024 * 1024)
                    seen_names.add(name)

                    if apply_aging:
                        with _first_seen_lock:                  # ← اصلاح (2a)
                            first_t = _first_seen_ts.setdefault(name, now)
                        wait_sec = max(0.0, now - first_t)
                        effective_bytes = max(0.0, sz_bytes - (wait_sec * _AGING_MB_PER_SEC * 1048576.0))
                    else:
                        effective_bytes = float(sz_bytes)

                    out.append({
                        "name": name,
                        "file": name,
                        "size_bytes": sz_bytes,
                        "size_mb": round(sz_mb, 2),
                        "_sort_key": effective_bytes,
                    })
                except OSError:
                    pass
    except OSError:
        return []

    if apply_aging:
        with _first_seen_lock:                                  # ← اصلاح (2b)
            for k in list(_first_seen_ts.keys()):
                if k not in seen_names:
                    _first_seen_ts.pop(k, None)

    out.sort(key=lambda x: (x["_sort_key"], x["size_bytes"], x["name"]))
    for item in out:
        item.pop("_sort_key", None)
    return out


# ==============================================================================
# SJF + Aging Background Ingestion Worker
# ==============================================================================
def _sjf_worker_loop() -> None:
    global _force_dir_rescan
    time.sleep(0.8)
    logger.info("SJF+Aging Background Worker started (watching inbox/).")
    while not _worker_stop.is_set():
        try:
            q_items = _scan_csv_dir(INBOX_DIR, apply_aging=True)
            if not q_items:
                _worker_stop.wait(0.35)
                continue

            chosen = None
            for candidate in q_items:
                cand_path = os.path.join(INBOX_DIR, candidate["name"])
                if _is_file_stable_and_ready(cand_path):
                    chosen = candidate
                    break

            if chosen is None:
                _worker_stop.wait(0.25)
                continue

            target_file = os.path.join(INBOX_DIR, chosen["name"])
            if not os.path.exists(target_file):
                continue

            logger.info(f"SJF picked: {chosen['name']} ({chosen['size_mb']} MB)")
            _force_dir_rescan = True
            try:
                res = engine.process_file(
                    target_file, ledger=ledger, profiler=profiler
                )
                ns, rows, raw_mb, comp_mb, elapsed = res
                with _first_seen_lock:                          # ← اصلاح (3a)
                    _first_seen_ts.pop(chosen["name"], None)
                _force_dir_rescan = True
                spd = raw_mb / max(0.001, elapsed)
                health = ledger.get_namespaces().get(ns, {}).get("kpis", {}).get("financial_status", "OK")  # ← اصلاح (3b)
                logger.info(
                    f"Completed {chosen['name']} -> [{ns}] | "
                    f"{rows:,} rows | {raw_mb:.1f}->{comp_mb:.1f} MB | "
                    f"{elapsed:.2f}s ({spd:.1f} MB/s) | Status: {health}"
                )
            except Exception as ex:
                logger.error(f"[SJF Worker] Error processing {chosen['name']}: {ex}")
                _force_dir_rescan = True
                if os.path.exists(target_file):
                    dst_fail = os.path.join(FAILED_INBOX_DIR, chosen["name"])
                    try:
                        if os.path.exists(dst_fail):
                            os.remove(dst_fail)
                        shutil.move(target_file, dst_fail)
                    except OSError:
                        pass
                _worker_stop.wait(0.5)
        except Exception as outer_ex:
            logger.error(f"SJF Worker loop error: {outer_ex}")
            _worker_stop.wait(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_thread
    _worker_stop.clear()
    _worker_thread = threading.Thread(target=_sjf_worker_loop, name="TitanSJFWorker", daemon=False)
    _worker_thread.start()
    yield
    logger.info("Initiating graceful shutdown of Titan Engine & SJF Worker...")
    _worker_stop.set()
    engine.request_stop()
    if _worker_thread is not None and _worker_thread.is_alive():
        _worker_thread.join(timeout=15.0)
    try:
        with engine._duck_lock:
            engine._duck_conn.close()
    except Exception:
        pass
    logger.info("Titan Server shutdown complete.")


app = FastAPI(title="Titan Telemetry & Financial KPI Factory v5.9.0-Enterprise", lifespan=lifespan)


# ==============================================================================
# Cached Dashboard State Snapshot (Zero Event-Loop Blocking & Zero-Seek on HDD)
# ==============================================================================
def _build_dashboard_state_sync() -> Dict[str, Any]:
    global _state_cache_ts, _state_cache_payload, _force_dir_rescan
    now = time.perf_counter()
    with _state_cache_lock:
        if (now - _state_cache_ts) < 0.25 and _state_cache_payload:
            return _state_cache_payload

    hw = profiler.sample()

    # Thread-safe copy of live_progress to avoid dictionary mutation races during streaming
    with engine._lock:
        prog = dict(engine.live_progress)

    is_busy = bool(getattr(engine, "_engine_busy", False) or prog.get("active", False))

    # ZERO-SEEK FAST PATH: Do not touch data-disk directories or SQLite while streaming 50GB files
    if is_busy and _state_cache_payload and not _force_dir_rescan:
        q_list = _state_cache_payload.get("queue", [])
        saved_list = _state_cache_payload.get("saved_files", [])
        namespaces = _state_cache_payload.get("namespaces", {})
        files_log = _state_cache_payload.get("files_log", [])
    else:
        q_list = _scan_csv_dir(INBOX_DIR, apply_aging=True)
        saved_list = _scan_csv_dir(SAVED_INBOX_DIR, apply_aging=False)
        namespaces = ledger.get_namespaces()
        files_log = ledger.get_recent_files(limit=25)
        _force_dir_rescan = False

    active_fname = str(prog.get("file") or "")
    pct_val = float(prog.get("pct") or 0.0)
    raw_mb_val = float(prog.get("raw_mb") or 0.0)
    done_mb_val = float(prog.get("done_mb") or 0.0)
    spd_val = float(prog.get("speed_mbps") or 0.0)
    eta_val = float(prog.get("eta_s") or 0.0)
    rows_val = int(prog.get("rows") or 0)

    if prog.get("active") and active_fname:
        prog[active_fname] = {
            "progress_pct": pct_val,
            "pct": pct_val,
            "processed_mb": done_mb_val,
            "done_mb": done_mb_val,
            "total_mb": raw_mb_val,
            "raw_mb": raw_mb_val,
            "speed_mbps": spd_val,
            "rows": rows_val,
            "eta_s": eta_val,
            "stage": prog.get("stage", "STREAMING"),
        }

    payload = {
        "hardware": hw,
        "hw": hw,
        "progress": prog,
        "live_progress": prog,
        "queue": q_list,
        "sjf_queue": q_list,
        "saved_files": saved_list,
        "namespaces": namespaces,
        "files_log": files_log,
        "recent_files": files_log,
        "ledger_version": ledger._version,
    }

    with _state_cache_lock:
        _state_cache_ts = time.perf_counter()
        _state_cache_payload = payload
    return payload


# ==============================================================================
# HTTP & WebSocket Endpoints
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    global _html_cache_mtime, _html_cache_content
    html_path = os.path.join(BASE_DIR, "index.html")
    # Avoid statting index.html on the data disk if engine is actively streaming and cache is warm
    if not (getattr(engine, "_engine_busy", False) and _html_cache_content):
        try:
            mtime = os.path.getmtime(html_path)
            if mtime != _html_cache_mtime or not _html_cache_content:
                with open(html_path, "r", encoding="utf-8") as f:
                    _html_cache_content = f.read()
                _html_cache_mtime = mtime
        except OSError:
            if not _html_cache_content:
                raise HTTPException(status_code=404, detail="index.html not found")

    return HTMLResponse(
        content=_html_cache_content,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/api/state")
async def api_state():
    payload = await asyncio.to_thread(_build_dashboard_state_sync)
    return JSONResponse(payload)


@app.get("/api/kpis")
async def api_all_kpis():
    return JSONResponse(engine.get_namespace_kpis())


@app.get("/api/kpis/{namespace}")
async def api_namespace_kpis(namespace: str):
    data = engine.get_namespace_kpis(namespace)
    if not data:
        raise HTTPException(status_code=404, detail=f"Namespace {namespace} not found")
    return JSONResponse(data)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while not _worker_stop.is_set():
            payload = await asyncio.to_thread(_build_dashboard_state_sync)
            await ws.send_json(payload)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/api/query/{namespace}")
async def api_query_namespace(
    namespace: str,
    min_val: Optional[float] = Query(default=None),
    filter_val: Optional[float] = Query(default=None),
):
    effective_filter = filter_val if filter_val is not None else min_val
    filter_key = repr(float(effective_filter)) if effective_filter is not None else "NONE"
    cache_key = (namespace, filter_key, ledger._version)

    with _query_cache_lock:
        if cache_key in _query_cache:
            return JSONResponse(_query_cache[cache_key])

    ns_map = ledger.get_namespaces()
    if namespace not in ns_map:
        return JSONResponse({
            "namespace": namespace,
            "status": "empty",
            "matched_rows": 0,
            "filtered_rows": 0,
            "top_categories": [],
            "buckets": [],
            "counts": [],
            "latency_ms": 0.0,
            "exec_ms": 0.0,
        })

    ns_info = ns_map[namespace]
    num_col = ns_info.get("num_col", "")
    cat_col = ns_info.get("cat_col", "")

    res = await asyncio.to_thread(
        engine.query_lakehouse,
        namespace,
        min_val=effective_filter,
        num_col=num_col,
        cat_col=cat_col,
    )

    # Populate buckets & counts for backward compatibility with all UI/test versions
    top_cats = res.get("top_categories", [])
    if "buckets" not in res:
        res["buckets"] = [str(item.get("category", "")) for item in top_cats]
    if "counts" not in res:
        res["counts"] = [int(item.get("count", 0)) for item in top_cats]

    if not res.get("engine_busy"):
        with _query_cache_lock:
            if len(_query_cache) > 256:
                _query_cache.clear()
            _query_cache[cache_key] = res

    return JSONResponse(res)


@app.post("/api/compact/{namespace}")
async def api_compact_namespace(namespace: str):
    res = await asyncio.to_thread(engine.compact_namespace, namespace, ledger)
    with _query_cache_lock:
        _query_cache.clear()
    return JSONResponse(res)


@app.post("/api/replay/{filename}")
async def api_replay_saved_file(filename: str):
    global _force_dir_rescan
    safe_name = os.path.basename(filename)
    src = os.path.join(SAVED_INBOX_DIR, safe_name)
    dst = os.path.join(INBOX_DIR, safe_name)

    if not os.path.exists(src):
        return JSONResponse(
            {"status": "ERROR", "detail": f"File {safe_name} not found in inbox_saved/"},
            status_code=404,
        )

    shutil.move(src, dst)
    _force_dir_rescan = True
    return JSONResponse({"status": "QUEUED", "file": safe_name})


# ==============================================================================
# High-Entropy Multi-Block Synthetic CSV Generator (Exact Byte Target)
# ==============================================================================
def _generate_csv_file(size_mb: int, schema_type: str) -> str:
    global _force_dir_rescan
    if size_mb <= 0 or size_mb > _MAX_GENERATE_MB:
        raise ValueError(f"size_mb must be between 1 and {_MAX_GENERATE_MB}")

    ts_ms = int(time.time() * 1000)
    tmp_name = f".gen_{schema_type}_{size_mb}MB_{ts_ms}.tmp"
    final_name = f"sim_{schema_type}_{size_mb}MB_{ts_ms}.csv"
    tmp_path = os.path.join(INBOX_DIR, tmp_name)
    final_path = os.path.join(INBOX_DIR, final_name)

    target_bytes = int(size_mb) * 1024 * 1024
    schema_type = schema_type.lower().strip()

    # Build 16 distinct randomized blocks to prevent unrealistic repetition
    blocks: List[bytes] = []
    if schema_type == "schema_b":
        header = b"ts,endpoint,latency_ms,http_code\n"
        categories = ["/api/v1/pay", "/api/v1/auth", "/api/v1/orders", "/api/v1/profile", "/api/v1/search", "/health"]
        for b in range(16):
            lines = [
                f"2026-03-30T{b % 24:02d}:{i % 60:02d}:{(i * 7 + b) % 60:02d}Z,"
                f"{random.choice(categories)},{round(random.lognormvariate(3.5, 0.75), 2)},200\n"
                for i in range(2500)
            ]
            blocks.append("".join(lines).encode("utf-8"))
    elif schema_type == "schema_c":
        # Signed PnL / Trade cashflow distribution (exercises Signed DDSketch, VaR, Sortino & Profit Factor)
        header = b"trade_ts,symbol,price_usd,exchange\n"
        categories = ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA", "AAPL", "MSFT", "AMZN", "META"]
        for b in range(16):
            lines = [
                f"2026-03-30T{b % 24:02d}:{i % 60:02d}:{(i * 11 + b) % 60:02d}Z,"
                f"{random.choice(categories)},{round(random.gauss(65.0, 95.0), 2)},NASDAQ\n"
                for i in range(2500)
            ]
            blocks.append("".join(lines).encode("utf-8"))
    else:
        header = b"timestamp,sensor_id,temperature,status\n"
        categories = [f"SNS_{i:04d}" for i in range(1, 2001)]
        statuses = ["OK", "OK", "OK", "WARN", "ALRT"]
        for b in range(16):
            lines = [
                f"2026-03-30 {b % 24:02d}:{i % 60:02d}:{(i * 13 + b) % 60:02d},"
                f"{random.choice(categories)},{round(random.gauss(72.0, 18.5), 2)},{random.choice(statuses)}\n"
                for i in range(2500)
            ]
            blocks.append("".join(lines).encode("utf-8"))

    with open(tmp_path, "wb", buffering=8 * 1024 * 1024) as f:
        f.write(header)
        written = len(header)
        while written < target_bytes:
            blk = random.choice(blocks)
            rem = target_bytes - written
            if len(blk) <= rem:
                f.write(blk)
                written += len(blk)
            else:
                nl_pos = blk.rfind(b"\n", 0, rem)
                if nl_pos > 0:
                    f.write(blk[: nl_pos + 1])
                break

    os.replace(tmp_path, final_path)
    _force_dir_rescan = True
    return final_name


@app.post("/api/generate/{size_mb}/{schema_type}")
@app.get("/api/generate/{size_mb}/{schema_type}")
async def api_generate(size_mb: int, schema_type: str):
    if size_mb <= 0 or size_mb > _MAX_GENERATE_MB:
        raise HTTPException(status_code=400, detail=f"size_mb must be between 1 and {_MAX_GENERATE_MB}")
    fname = await asyncio.to_thread(_generate_csv_file, size_mb, schema_type)
    return JSONResponse({"status": "GENERATED", "file": fname, "size_mb": size_mb})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
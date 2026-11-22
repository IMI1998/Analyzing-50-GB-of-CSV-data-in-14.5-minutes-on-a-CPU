# engine.py
# Titan Data Engine v6.0.0 — Handle-Safe Atomic Finalize + Drain-Mode Workers + Memory Governor
import io
import os
import re
import time
import math
import queue
import threading
import logging
from typing import Optional, Dict, Any, List, Tuple

import duckdb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.compute as pc
import pyarrow.parquet as pq

from telemetry_math import (
    ZERO_EPSILON,
    INV_GAMMA_LN,
    HLL_M,
    DD_OFFSET,
    DD_SPAN,
    DD_EPS_SLOT,
    MergeableSketchState,
    ZeroOverheadProfiler,
    ManifestLedger,
    hll_update_from_arrow,
)

logger = logging.getLogger("TitanEngine")

# ==============================================================================
# Directory & Hardware Constants (UNCHANGED tuned values)
# ==============================================================================
BASE_DIR = os.environ.get("TITAN_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
INBOX_DIR = os.path.join(BASE_DIR, "inbox")
SAVED_INBOX_DIR = os.path.join(BASE_DIR, "inbox_saved")
FAILED_INBOX_DIR = os.path.join(BASE_DIR, "inbox_failed")
QUARANTINE_DIR = os.path.join(BASE_DIR, "quarantine")
LAKEHOUSE_DIR = os.path.join(BASE_DIR, "lakehouse")

_LOGICAL_CPUS = os.cpu_count() or 4
pa.set_cpu_count(max(2, _LOGICAL_CPUS // 2))
pa.set_io_thread_count(2)

CSV_BLOCK_SIZE = 32 * 1024 * 1024
_CHUNK_SIZE: int = 262144
_QUEUE_DEPTH: int = 16
_TARGET_RG_ROWS: int = 1_500_000
_MB: float = 1048576.0

# Memory-pressure governor: inactive during normal flow -> zero throughput cost
_SEQUENTIAL_SCAN: bool = os.environ.get("TITAN_SEQ_SCAN", "1") == "1"
_TOTAL_RAM = psutil.virtual_memory().total
_ARROW_BUDGET_BYTES: int = int(os.environ.get(
    "TITAN_ARROW_BUDGET_MB", str(min(4096, int(_TOTAL_RAM * 0.28 / _MB)))
)) * 1048576
_MIN_AVAIL_GB: float = float(os.environ.get("TITAN_MIN_AVAIL_GB", "1.0"))

_NUMERIC_TYPES = {
    pa.int8(), pa.int16(), pa.int32(), pa.int64(),
    pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
    pa.float16(), pa.float32(), pa.float64(),
}
_STRING_TYPES = {pa.string(), pa.large_string(), pa.utf8(), pa.large_utf8()}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Interleaved DDSketch slot layout: slot = 2*(k+DD_OFFSET) + is_negative; last slot = zero bucket
_N_SLOTS: int = 2 * DD_SPAN + 1
_ZERO_SLOT: int = 2 * DD_SPAN
_SLOT_OFF2: float = float(2 * DD_OFFSET)
_INV2: float = 2.0 * INV_GAMMA_LN  # unused fast constant kept for clarity of layout math

# ------------------------------------------------------------------------------
# Allocator: system pool returns freed pages to the OS (cross-thread free safe)
# ------------------------------------------------------------------------------
if os.environ.get("TITAN_ARROW_POOL", "system") == "system":
    try:
        pa.set_memory_pool(pa.system_memory_pool())
    except Exception:
        pass


def _release_arrow_memory() -> None:
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def _ensure_directories(base_dir: str = BASE_DIR) -> None:
    for sub in ("", "inbox", "inbox_saved", "inbox_failed", "quarantine", "lakehouse"):
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)


_ensure_directories(BASE_DIR)


class _Neumaier:
    """Compensated summation across batches (error O(eps) independent of batch count)."""
    __slots__ = ("s", "c")

    def __init__(self) -> None:
        self.s = 0.0
        self.c = 0.0

    def add(self, x: float) -> None:
        t = self.s + x
        if abs(self.s) >= abs(x):
            self.c += (self.s - t) + x
        else:
            self.c += (x - t) + self.s
        self.s = t

    @property
    def value(self) -> float:
        return self.s + self.c


# ==============================================================================
# L3-Resident Accumulator
# ==============================================================================
class _FastNativeAccumulator:
    __slots__ = (
        "count", "mean", "m2", "min_val", "max_val",
        "pos_sum", "neg_sum", "downside_m2", "pos_count", "neg_count", "zero_count",
        "slots", "hll_regs", "buf_a", "buf_b", "buf_i32",
    )

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.pos_sum = _Neumaier()
        self.neg_sum = _Neumaier()
        self.downside_m2 = _Neumaier()
        self.pos_count = 0
        self.neg_count = 0
        self.zero_count = 0
        self.slots = np.zeros(_N_SLOTS, dtype=np.int64)
        self.hll_regs = np.zeros(HLL_M, dtype=np.uint8)
        self.buf_a = np.empty(_CHUNK_SIZE, dtype=np.float64)
        self.buf_b = np.empty(_CHUNK_SIZE, dtype=np.float64)
        self.buf_i32 = np.empty(_CHUNK_SIZE, dtype=np.int32)

    def merge_moments(self, n: int, mean: float, m2: float, mn: float, mx: float) -> None:
        if self.count == 0:
            self.count, self.mean, self.m2, self.min_val, self.max_val = n, mean, m2, mn, mx
            return
        tot = self.count + n
        delta = mean - self.mean
        self.mean += delta * (n / tot)
        self.m2 += m2 + delta * delta * (self.count * (n / tot))
        self.min_val = min(self.min_val, mn)
        self.max_val = max(self.max_val, mx)
        self.count = tot

    def to_mergeable_sketch_state(self) -> MergeableSketchState:
        st = MergeableSketchState()
        if self.count > 0:
            body = self.slots[:_ZERO_SLOT]
            pos = body[0::2]
            neg = body[1::2]
            pos_d = {int(i - DD_OFFSET): int(pos[i]) for i in np.nonzero(pos)[0]}
            neg_d = {int(i - DD_OFFSET): int(neg[i]) for i in np.nonzero(neg)[0]}
            st.merge_batch_stats(
                b_n=self.count, b_min=self.min_val, b_max=self.max_val,
                b_mean=self.mean, b_m2=self.m2, b_bins=pos_d, b_neg_bins=neg_d,
                b_zero_count=int(self.slots[_ZERO_SLOT]),
                b_pos_sum=self.pos_sum.value, b_neg_sum=self.neg_sum.value,
                b_downside_m2=self.downside_m2.value,
                b_pos_count=self.pos_count, b_neg_count=self.neg_count,
            )
        st.hll_registers = self.hll_regs.copy()
        return st


def _kernel(arr: np.ndarray, acc: _FastNativeAccumulator) -> None:
    """Single fused pass per 2MB chunk: corrected two-pass m2 + signed log-binning + sums."""
    n = arr.shape[0]
    if n == 0:
        return
    b_mean = float(np.sum(arr)) / n
    b_min = float(np.min(arr))
    b_max = float(np.max(arr))
    all_pos = b_min > ZERO_EPSILON
    sq = 0.0
    lin = 0.0
    slots = acc.slots
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(0, n, _CHUNK_SIZE):
            sub = arr[i:i + _CHUNK_SIZE]
            m = sub.shape[0]
            a = acc.buf_a[:m]
            ib = acc.buf_i32[:m]
            np.subtract(sub, b_mean, out=a)
            lin += float(np.sum(a))
            sq += float(np.dot(a, a))
            if all_pos:
                np.log(sub, out=a)
                np.multiply(a, INV_GAMMA_LN, out=a)
                np.ceil(a, out=a)
                np.multiply(a, 2.0, out=a)
                np.add(a, _SLOT_OFF2, out=a)
                np.copyto(ib, a, casting="unsafe")
                slots += np.bincount(ib, minlength=_N_SLOTS)[:_N_SLOTS]
                acc.pos_sum.add(float(np.sum(sub)))
                acc.pos_count += m
                continue
            b = acc.buf_b[:m]
            np.abs(sub, out=b)
            zmask = b <= ZERO_EPSILON
            negmask = sub < -ZERO_EPSILON
            posmask = sub > ZERO_EPSILON
            np.log(b, out=a)
            np.multiply(a, INV_GAMMA_LN, out=a)
            np.ceil(a, out=a)
            np.multiply(a, 2.0, out=a)
            np.add(a, _SLOT_OFF2, out=a)
            np.add(a, negmask, out=a)
            np.copyto(ib, a, casting="unsafe")
            np.putmask(ib, zmask, _ZERO_SLOT)
            slots += np.bincount(ib, minlength=_N_SLOTS)[:_N_SLOTS]
            pc_ = int(np.count_nonzero(posmask))
            nc_ = int(np.count_nonzero(negmask))
            acc.pos_count += pc_
            acc.neg_count += nc_
            acc.zero_count += m - pc_ - nc_
            if pc_:
                acc.pos_sum.add(float(np.sum(sub, where=posmask)))
            if nc_:
                acc.neg_sum.add(-float(np.sum(sub, where=negmask)))
                np.multiply(sub, sub, out=b)
                acc.downside_m2.add(float(np.sum(b, where=negmask)))
    b_m2 = max(0.0, sq - lin * lin / n)  # Björck corrected two-pass
    acc.merge_moments(n, b_mean, b_m2, b_min, b_max)


def _update_accumulator_peak(num_c: Optional[pa.Array], cat_c: Optional[pa.Array],
                             acc: _FastNativeAccumulator) -> None:
    if num_c is not None and len(num_c) > 0:
        col = num_c if num_c.null_count == 0 else pc.drop_null(num_c)
        if len(col) > 0:
            arr = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)
            s = float(np.sum(arr))
            if not math.isfinite(s):  # NaN/inf present OR overflow -> exact filtered path
                arr = arr[np.isfinite(arr)]
            _kernel(arr, acc)
    if cat_c is not None:
        hll_update_from_arrow(acc.hll_regs, cat_c)


# ==============================================================================
# Titan Engine
# ==============================================================================
class TitanEngine:
    def __init__(self, profiler: Optional[ZeroOverheadProfiler] = None,
                 ledger: Optional[ManifestLedger] = None, base_dir: str = BASE_DIR):
        self.base_dir = base_dir
        _ensure_directories(self.base_dir)
        self.profiler = profiler or ZeroOverheadProfiler()
        self.ledger = ledger or ManifestLedger(os.path.join(self.base_dir, "manifest.db"))
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._engine_busy = False
        self._duck_lock = threading.Lock()
        self._duck_conn = duckdb.connect(database=":memory:")
        self._duck_conn.execute(f"PRAGMA threads={max(2, _LOGICAL_CPUS)};")
        self._duck_conn.execute("PRAGMA memory_limit='2GB';")
        self.live_progress: Dict[str, Any] = {
            "active": False, "file": None, "namespace": None, "rows": 0, "raw_mb": 0.0,
            "done_mb": 0.0, "pct": 0.0, "speed_mbps": 0.0, "eta_s": 0.0, "stage": "IDLE",
        }

    def request_stop(self) -> None:
        self._stop_event.set()

    # --------------------------------------------------------------------------
    def _sniff_schema_and_namespace(self, csv_path: str, active_ledger: ManifestLedger
                                    ) -> Tuple[str, str, str, Dict[str, pa.DataType]]:
        with open(csv_path, "rb") as f:
            sample = f.read(65536)
        if not sample.strip():
            raise ValueError(f"Empty or unreadable CSV file: {csv_path}")
        nl = sample.rfind(b"\n")
        if nl > 0 and len(sample) == 65536:
            sample = sample[:nl]
        rd = pv.open_csv(pa.py_buffer(sample),
                         read_options=pv.ReadOptions(block_size=max(65536, len(sample) + 1024)))
        first = rd.read_next_batch()
        schema = first.schema
        cols = list(schema.names)
        typ = {c: schema.field(c).type for c in cols}

        num_col = ""
        for p in ("temperature", "latency_ms", "price_usd", "value", "amount", "metric", "score", "pnl"):
            num_col = next((c for c in cols if c.lower() == p and typ[c] in _NUMERIC_TYPES), "")
            if num_col:
                break
        if not num_col:
            num_col = next((c for c in cols if pa.types.is_floating(typ[c])), "")
        if not num_col:
            num_col = next((c for c in cols if typ[c] in _NUMERIC_TYPES
                            and not any(k in c.lower() for k in ("id", "ts", "time", "code"))), "")
        if not num_col:
            num_col = next((c for c in cols if typ[c] in _NUMERIC_TYPES), "")

        pref_cat = ("machine_line", "endpoint", "symbol", "sensor_id", "category", "device", "host", "status", "exchange")
        cat_col = ""
        for p in pref_cat:
            cat_col = next((c for c in cols if c.lower() == p and c != num_col and typ[c] in _STRING_TYPES), "")
            if cat_col:
                break
        if not cat_col:
            cat_col = next((c for c in cols if c != num_col and typ[c] in _STRING_TYPES
                            and not any(k in c.lower() for k in ("ts", "time", "date"))), "")
        if not cat_col:
            for p in pref_cat:
                cat_col = next((c for c in cols if c.lower() == p and c != num_col), "")
                if cat_col:
                    break
        if not cat_col:
            cat_col = next((c for c in cols if c != num_col), "")

        existing = active_ledger.get_namespaces()
        matched = next((ns for ns, info in existing.items()
                        if info.get("columns") == cols
                        or (num_col and info.get("num_col") == num_col and info.get("cat_col") == cat_col)), None)
        if matched is None:
            matched = f"schema_ns_{len(existing) + 1:02d}"

        overrides: Dict[str, pa.DataType] = {}
        n_s = max(1, len(first))
        for c in cols:
            t = typ[c]
            if c == num_col and t in _NUMERIC_TYPES:
                overrides[c] = pa.float64()
            elif t in _STRING_TYPES:
                ratio = len(pc.unique(first.column(c))) / n_s
                overrides[c] = pa.dictionary(pa.int32(), pa.string()) if (ratio < 0.5 or c == cat_col) else t
            else:
                overrides[c] = t
        return matched, num_col, cat_col, overrides

    # --------------------------------------------------------------------------
    # Drain-mode workers: after error/stop they keep CONSUMING until sentinel -> no deadlock
    # --------------------------------------------------------------------------
    def _parquet_writer_worker(self, write_q: "queue.Queue", tmp_path: str,
                               errors: List[BaseException], done_ctr: List[int]) -> None:
        writer: Optional[pq.ParquetWriter] = None
        buf: List[pa.RecordBatch] = []
        rows = 0
        failed = False
        try:
            while True:
                batch = write_q.get()
                if batch is None:
                    break
                if failed or self._stop_event.is_set() or errors:
                    failed = True
                    del batch
                    continue
                try:
                    if writer is None:
                        writer = pq.ParquetWriter(
                            tmp_path, batch.schema, compression="ZSTD", compression_level=1,
                            use_dictionary=True, write_statistics=True,
                            dictionary_pagesize_limit=8 * 1024 * 1024,
                            data_page_size=2 * 1024 * 1024, version="2.6",
                        )
                    buf.append(batch)
                    rows += batch.num_rows
                    del batch
                    if rows >= _TARGET_RG_ROWS:
                        tbl = pa.Table.from_batches(buf)
                        buf.clear()
                        writer.write_table(tbl, row_group_size=rows)
                        del tbl
                        rows = 0
                        _release_arrow_memory()
                    done_ctr[0] += 1
                except BaseException as ex:
                    errors.append(ex)
                    failed = True
                    buf.clear()
            if buf and writer is not None and not failed and not self._stop_event.is_set():
                writer.write_table(pa.Table.from_batches(buf), row_group_size=rows)
        except BaseException as ex:
            errors.append(ex)
        finally:
            buf.clear()
            if writer is not None:
                try:
                    writer.close()
                except Exception as ex:
                    errors.append(ex)

    def _stats_worker(self, stats_q: "queue.Queue", acc: _FastNativeAccumulator,
                      errors: List[BaseException]) -> None:
        failed = False
        while True:
            item = stats_q.get()
            if item is None:
                return
            if failed or self._stop_event.is_set() or errors:
                failed = True
                del item
                continue
            try:
                _update_accumulator_peak(item[0], item[1], acc)
            except BaseException as ex:
                errors.append(ex)
                failed = True
            del item

    # --------------------------------------------------------------------------
    def _open_source(self, path: str):
        if _SEQUENTIAL_SCAN and os.name == "nt" and hasattr(os, "O_SEQUENTIAL"):
            fd = os.open(path, os.O_RDONLY | os.O_BINARY | os.O_SEQUENTIAL)
            raw = os.fdopen(fd, "rb", buffering=0)
            return pa.PythonFile(raw, mode="r"), raw
        f = pa.OSFile(path, "r")
        return f, f

    def _memory_governor(self) -> None:
        """Backpressure only when the budget is exceeded; no-op in the normal case."""
        pool = pa.default_memory_pool()
        spins = 0
        while not self._stop_event.is_set() and spins < 400:
            if pool.bytes_allocated() < _ARROW_BUDGET_BYTES and \
               psutil.virtual_memory().available / 1073741824.0 > _MIN_AVAIL_GB:
                return
            if spins == 0:
                _release_arrow_memory()
            time.sleep(0.025)
            spins += 1

    def _set_progress(self, **kw: Any) -> None:
        with self._lock:
            self.live_progress.update(kw)

    # --------------------------------------------------------------------------
    def process_file(self, filepath: str, ledger: Optional[ManifestLedger] = None,
                     profiler: Optional[ZeroOverheadProfiler] = None) -> Tuple[str, int, float, float, float]:
        active_ledger = ledger or self.ledger
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        orig = os.path.basename(filepath)
        raw_bytes = os.path.getsize(filepath)
        raw_mb = raw_bytes / _MB
        claim = filepath + ".claiming"
        os.replace(filepath, claim)
        t0 = time.perf_counter()
        self._engine_busy = True
        self._set_progress(active=True, file=orig, namespace="DETECTING...", rows=0,
                           raw_mb=round(raw_mb, 2), done_mb=0.0, pct=0.0, speed_mbps=0.0,
                           eta_s=0.0, stage="SNIFFING_SCHEMA")

        src = handle = None
        reader = None
        write_q: "queue.Queue" = queue.Queue(maxsize=_QUEUE_DEPTH)
        stats_q: "queue.Queue" = queue.Queue(maxsize=_QUEUE_DEPTH)
        threads: List[threading.Thread] = []
        errors: List[BaseException] = []
        tmp_path = ""
        try:
            namespace, num_col, cat_col, overrides = self._sniff_schema_and_namespace(claim, active_ledger)
            ns_dir = os.path.join(self.base_dir, "lakehouse", namespace)
            os.makedirs(ns_dir, exist_ok=True)
            out_name = os.path.splitext(orig)[0] + f"_{int(time.time() * 1000)}.parquet"
            tmp_path = os.path.join(ns_dir, "." + out_name + ".tmp")
            final_path = os.path.join(ns_dir, out_name)
            self._set_progress(namespace=namespace, stage="STREAMING")

            acc = _FastNativeAccumulator()
            done_ctr = [0]
            threads = [
                threading.Thread(target=self._parquet_writer_worker,
                                 args=(write_q, tmp_path, errors, done_ctr), name="TitanParquetWriter", daemon=True),
                threading.Thread(target=self._stats_worker, args=(stats_q, acc, errors),
                                 name="TitanStatsWorker", daemon=True),
            ]
            for t in threads:
                t.start()

            src, handle = self._open_source(claim)
            reader = pv.open_csv(
                src,
                read_options=pv.ReadOptions(block_size=CSV_BLOCK_SIZE, use_threads=True),
                parse_options=pv.ParseOptions(invalid_row_handler=lambda row: "skip"),
                convert_options=pv.ConvertOptions(column_types=overrides, strings_can_be_null=True),
            )
            total_rows = batches = 0
            num_idx = cat_idx = -1
            names: List[str] = []
            for batch in reader:
                if self._stop_event.is_set() or errors:
                    break
                if not names:
                    names = list(batch.schema.names)
                    num_idx = names.index(num_col) if num_col in names else -1
                    cat_idx = names.index(cat_col) if cat_col in names else -1
                total_rows += batch.num_rows
                batches += 1
                num_c = batch.column(num_idx) if num_idx >= 0 else None
                cat_c = batch.column(cat_idx) if cat_idx >= 0 else None
                write_q.put(batch)
                stats_q.put((num_c, cat_c))
                del batch, num_c, cat_c
                if (batches & 7) == 0:
                    self._memory_governor()
                try:
                    pos = min(raw_bytes, int(src.tell()))
                except Exception:
                    pos = min(raw_bytes, batches * CSV_BLOCK_SIZE)
                el = max(0.001, time.perf_counter() - t0)
                done = pos / _MB
                spd = done / el
                self._set_progress(rows=total_rows, done_mb=round(done, 2),
                                   pct=round(min(97.0, 97.0 * pos / max(1, raw_bytes)), 1),
                                   speed_mbps=round(spd, 2),
                                   eta_s=round(max(0.0, raw_mb - done) / max(0.01, spd), 1))

            # Release the source handle BEFORE any rename (fixes Windows copy-fallback at 99.5%)
            reader = None
            try:
                src.close()
            finally:
                if handle is not src:
                    handle.close()
                src = handle = None
        finally:
            # Sentinels are ALWAYS delivered -> workers always terminate, buffers always freed
            if threads:
                write_q.put(None)
                stats_q.put(None)
                self._set_progress(stage="DRAINING")
                while any(t.is_alive() for t in threads):
                    threads[0].join(timeout=0.2)
                    b = max(1, locals().get("batches", 1))
                    self._set_progress(pct=round(97.0 + 2.5 * min(1.0, done_ctr[0] / b), 1))
                for t in threads:
                    t.join()
            if reader is not None:
                reader = None
            for h in (src, handle):
                try:
                    if h is not None:
                        h.close()
                except Exception:
                    pass

        try:
            if errors:
                raise errors[0]
            if self._stop_event.is_set():
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                os.replace(claim, filepath)
                raise InterruptedError("Ingestion aborted by graceful shutdown request.")

            self._set_progress(stage="FINALIZING", pct=99.6)
            os.replace(tmp_path, final_path)
            comp_mb = os.path.getsize(final_path) / _MB
            saved_dst = os.path.join(self.base_dir, "inbox_saved", orig)
            os.replace(claim, saved_dst)  # same volume -> O(1) metadata rename, never a copy

            elapsed = max(0.001, time.perf_counter() - t0)
            sketch = acc.to_mergeable_sketch_state()
            del acc
            _release_arrow_memory()
            active_ledger.commit_file_ingestion(
                namespace=namespace, filename=orig, parquet_file=out_name, rows=total_rows,
                raw_mb=raw_mb, comp_mb=comp_mb, elapsed_s=elapsed,
                num_col=num_col, cat_col=cat_col, sketch_state=sketch,
            )
            self._set_progress(active=False, rows=total_rows, done_mb=round(raw_mb, 2), pct=100.0,
                               speed_mbps=round(raw_mb / elapsed, 2), eta_s=0.0, stage="COMPLETED")
            return namespace, total_rows, raw_mb, comp_mb, elapsed
        except InterruptedError:
            self._set_progress(active=False, stage="ABORTED", eta_s=0.0)
            raise
        except Exception as ex:
            logger.error(f"Ingestion failed for {orig}: {ex}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if os.path.exists(claim):
                try:
                    os.replace(claim, os.path.join(self.base_dir, "inbox_failed", orig))
                except OSError:
                    pass
            self._set_progress(active=False, stage="FAILED", eta_s=0.0)
            raise
        finally:
            self._engine_busy = False
            _release_arrow_memory()

    # --------------------------------------------------------------------------
    def get_namespace_kpis(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        ns_map = self.ledger.get_namespaces()
        if namespace:
            info = ns_map.get(namespace)
            return info.get("kpis", {}) if info else {}
        return {ns: d.get("kpis", {}) for ns, d in ns_map.items()}

    def query_lakehouse(self, namespace: str, min_val: Optional[float] = None,
                        num_col: str = "", cat_col: str = "") -> Dict[str, Any]:
        t0 = time.perf_counter()
        ns_map = self.ledger.get_namespaces()
        info = ns_map.get(namespace, {})
        kpis = info.get("kpis", {})

        def _empty(status: str, busy: bool = False) -> Dict[str, Any]:
            ms = round((time.perf_counter() - t0) * 1000.0, 2)
            r = int(info.get("total_rows", 0)) if busy else 0
            return {"namespace": namespace, "status": status, "engine_busy": busy,
                    "matched_rows": r, "filtered_rows": r, "top_categories": [],
                    "buckets": [], "counts": [], "kpis": kpis, "latency_ms": ms, "exec_ms": ms}

        if self._engine_busy:
            return _empty("busy_streaming", True)
        ns_dir = os.path.join(self.base_dir, "lakehouse", namespace)
        if not os.path.isdir(ns_dir) or not any(
                f.endswith(".parquet") and not f.startswith(".") for f in os.listdir(ns_dir)):
            return _empty("empty")
        num_col = num_col or info.get("num_col", "")
        cat_col = cat_col or info.get("cat_col", "")
        for ident in (num_col, cat_col):
            if ident and not _IDENTIFIER_RE.match(ident):
                raise ValueError(f"Unsafe SQL identifier rejected: {ident}")
        glob = os.path.join(ns_dir, "[!.]*.parquet").replace("\\", "/")
        with self._duck_lock:
            if min_val is not None and num_col and cat_col:
                rows = self._duck_conn.execute(
                    f'SELECT CAST("{cat_col}" AS VARCHAR), COUNT(*) c FROM read_parquet(\'{glob}\') '
                    f'WHERE "{num_col}" >= ? GROUP BY 1 ORDER BY c DESC LIMIT 12', [float(min_val)]).fetchall()
                total = int(self._duck_conn.execute(
                    f'SELECT COUNT(*) FROM read_parquet(\'{glob}\') WHERE "{num_col}" >= ?',
                    [float(min_val)]).fetchone()[0])
            elif cat_col:
                rows = self._duck_conn.execute(
                    f'SELECT CAST("{cat_col}" AS VARCHAR), COUNT(*) c FROM read_parquet(\'{glob}\') '
                    f'GROUP BY 1 ORDER BY c DESC LIMIT 12').fetchall()
                total = int(info.get("total_rows", sum(r[1] for r in rows)))
            else:
                rows, total = [], int(info.get("total_rows", 0))
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        tops = [{"category": str(r[0]), "count": int(r[1])} for r in rows]
        return {"namespace": namespace, "status": "ok", "matched_rows": total, "filtered_rows": total,
                "top_categories": tops, "buckets": [c["category"] for c in tops],
                "counts": [c["count"] for c in tops], "kpis": kpis, "latency_ms": ms, "exec_ms": ms}

    def compact_namespace(self, namespace: str, ledger: Optional[ManifestLedger] = None) -> Dict[str, Any]:
        if self._engine_busy:
            return {"status": "BUSY", "detail": "Cannot compact while streaming ingestion is active."}
        active_ledger = ledger or self.ledger
        ns_dir = os.path.join(self.base_dir, "lakehouse", namespace)
        if not os.path.isdir(ns_dir):
            return {"status": "SKIPPED", "detail": "Namespace directory does not exist."}
        files = sorted(os.path.join(ns_dir, f) for f in os.listdir(ns_dir)
                       if f.endswith(".parquet") and not f.startswith("."))
        if len(files) <= 1:
            return {"status": "SKIPPED", "detail": "Already compacted (<= 1 Parquet file).", "files_merged": 0}
        t0 = time.perf_counter()
        name = f"compacted_{int(time.time() * 1000)}.parquet"
        tmp = os.path.join(ns_dir, "." + name + ".tmp")
        final = os.path.join(ns_dir, name)
        writer: Optional[pq.ParquetWriter] = None
        total = 0
        try:
            for fp in files:
                pf = pq.ParquetFile(fp)
                try:
                    for rg in range(pf.num_row_groups):
                        tbl = pf.read_row_group(rg)
                        if writer is None:
                            writer = pq.ParquetWriter(tmp, tbl.schema, compression="ZSTD", compression_level=1,
                                                      use_dictionary=True, write_statistics=True, version="2.6")
                        writer.write_table(tbl, row_group_size=_TARGET_RG_ROWS)
                        total += tbl.num_rows
                        del tbl
                finally:
                    pf.close()  # release handle so os.remove works on Windows
            writer.close()
            writer = None
        except BaseException:
            if writer is not None:
                writer.close()
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        os.replace(tmp, final)
        for fp in files:
            try:
                os.remove(fp)
            except OSError:
                pass
        try:
            active_ledger.record_compaction(namespace, old_count=len(files), new_count=1)
        except Exception:
            pass
        _release_arrow_memory()
        return {"status": "COMPACTED", "namespace": namespace, "files_merged": len(files),
                "total_rows": total, "new_size_mb": round(os.path.getsize(final) / _MB, 2),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
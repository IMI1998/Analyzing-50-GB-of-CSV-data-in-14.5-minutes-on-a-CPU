# telemetry_math.py
# Titan Mathematical & Transactional Persistence Core v6.0.0
# Exhaustive-domain Signed DDSketch + Ertl-HLL (deterministic hash) + Corrected Chan-Welford
from __future__ import annotations

import os
import sys
import json
import math
import time
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc

# ==============================================================================
# 1. IEEE-754 CONSTANTS
# ==============================================================================
DDSKETCH_ALPHA: float = 0.005
GAMMA: float = (1.0 + DDSKETCH_ALPHA) / (1.0 - DDSKETCH_ALPHA)
GAMMA_LN: float = math.log(GAMMA)
INV_GAMMA_LN: float = 1.0 / GAMMA_LN
_INV_GAMMA_LN: float = INV_GAMMA_LN  # backward-compat alias
ZERO_EPSILON: float = 1e-12

_LN_MIDPOINT: float = math.log(2.0 / (1.0 + GAMMA))
_LN_FLOAT_MAX: float = math.log(sys.float_info.max)


def _vectorized_bin_index(x: float) -> int:
    """Bin index computed through the SAME vectorized numpy path the engine kernels use."""
    a = np.full(16, x, dtype=np.float64)
    np.log(a, out=a)
    np.multiply(a, INV_GAMMA_LN, out=a)
    np.ceil(a, out=a)
    return int(a[0])


# Exhaustive bin domain: every finite |x| in (ZERO_EPSILON, DBL_MAX] has a slot -> no clipping, ever.
DD_K_EPS: int = _vectorized_bin_index(ZERO_EPSILON)
DD_K_MIN: int = DD_K_EPS - 2
DD_K_MAX: int = _vectorized_bin_index(sys.float_info.max) + 2
DD_OFFSET: int = -DD_K_MIN
DD_SPAN: int = DD_K_MAX - DD_K_MIN + 1
DD_EPS_SLOT: int = 2 * (DD_K_EPS + DD_OFFSET)  # interleaved (even = positive) slot of ZERO_EPSILON

HLL_P: int = 12
HLL_M: int = 1 << HLL_P
HLL_Q: int = 64 - HLL_P          # max rho = HLL_Q + 1 = 53
HLL_VERSION: int = 2
_HLL_SHIFT = np.uint64(HLL_Q)
_HLL_LSHIFT = np.uint64(HLL_P)
_HLL_SENTINEL = np.uint64(1 << (HLL_P - 1))
_U32_MASK = np.uint64(0xFFFFFFFF)
_U32_SHIFT = np.uint64(32)

# SplitMix64 finalizer constants (Steele/Lea/Flood) – bijective full-avalanche mixer
_SM_C1 = np.uint64(0xBF58476D1CE4E5B9)
_SM_C2 = np.uint64(0x94D049BB133111EB)
_SM_S30, _SM_S27, _SM_S31 = np.uint64(30), np.uint64(27), np.uint64(31)
_SEED_BYTES = np.uint64(0x9E3779B97F4A7C15)
_SEED_LEN = np.uint64(0xD6E8FEB86659FD93)
_SEED_INT = np.uint64(0xA0761D6478BD642F)
_SEED_FLT = np.uint64(0xE7037ED1A0B428DB)
_U8_SHIFT = np.uint64(8)


def _bin_to_value(k: int) -> float:
    """Relative-error-optimal representative 2*gamma^k/(1+gamma), computed in log space."""
    ln_v = k * GAMMA_LN + _LN_MIDPOINT
    if ln_v >= _LN_FLOAT_MAX:
        return sys.float_info.max
    return math.exp(ln_v)


# ==============================================================================
# 2. DETERMINISTIC VECTORIZED HASHING + ERTL HYPERLOGLOG
# ==============================================================================
def _splitmix64_inplace(x: np.ndarray) -> np.ndarray:
    x ^= x >> _SM_S30
    x *= _SM_C1
    x ^= x >> _SM_S27
    x *= _SM_C2
    x ^= x >> _SM_S31
    return x


def _hash_bytes_array(arr: pa.Array, off_dtype: type) -> np.ndarray:
    """
    Additive tabulation hash over (position, byte) pairs, fully vectorized:
      h(s) = mix( sum_i T[i, s_i]  XOR  len*C ),  T[i,b] = splitmix64((i<<8 | b) ^ seed)
    Deterministic across processes (unlike Python's salted hash()) and immune to the
    Thue-Morse collision family that breaks polynomial hashing mod 2^64.
    """
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    bufs = arr.buffers()
    itemsize = np.dtype(off_dtype).itemsize
    offs = np.frombuffer(bufs[1], dtype=off_dtype, count=n + 1, offset=arr.offset * itemsize).astype(np.int64)
    base = int(offs[0])
    total = int(offs[-1]) - base
    lengths = np.diff(offs)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        if total > 0 and bufs[2] is not None:
            data = np.frombuffer(bufs[2], dtype=np.uint8, count=total, offset=base)
            starts = offs[:-1] - base
            sid = np.repeat(np.arange(n, dtype=np.int64), lengths)
            pos = np.arange(total, dtype=np.int64) - starts[sid]
            key = (pos.astype(np.uint64) << _U8_SHIFT) | data.astype(np.uint64)
            key ^= _SEED_BYTES
            _splitmix64_inplace(key)
            nz = lengths > 0
            h[nz] = np.add.reduceat(key, starts[nz])
        h ^= lengths.astype(np.uint64) * _SEED_LEN
        return _splitmix64_inplace(h)


def _hash_arrow_values(arr: pa.Array) -> Optional[np.ndarray]:
    t = arr.type
    with np.errstate(over="ignore"):
        if pa.types.is_string(t) or pa.types.is_binary(t):
            return _hash_bytes_array(arr, np.int32)
        if pa.types.is_large_string(t) or pa.types.is_large_binary(t):
            return _hash_bytes_array(arr, np.int64)
        if pa.types.is_floating(t):
            v = np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.float64) + 0.0  # -0.0 -> +0.0
            u = v.view(np.uint64)
            u ^= _SEED_FLT
            return _splitmix64_inplace(u)
        if pa.types.is_integer(t):
            v = pc.cast(arr, pa.int64(), safe=False).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
            u = v.view(np.uint64)
            u ^= _SEED_INT
            return _splitmix64_inplace(u)
    try:
        return _hash_bytes_array(pc.cast(arr, pa.string()), np.int32)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return None


def hll_update_from_arrow(regs: np.ndarray, arr: Optional[pa.Array]) -> None:
    """Idempotent HLL update. Only distinct values are hashed (dictionary or pc.unique)."""
    if arr is None or len(arr) == 0:
        return
    if isinstance(arr, pa.ChunkedArray):
        for ch in arr.chunks:
            hll_update_from_arrow(regs, ch)
        return
    if pa.types.is_dictionary(arr.type):
        if arr.null_count == len(arr):
            return
        uniq = arr.dictionary
    else:
        uniq = pc.unique(arr)
    if uniq.null_count:
        uniq = pc.drop_null(uniq)
    if len(uniq) == 0:
        return
    h = _hash_arrow_values(uniq)
    if h is None or h.size == 0:
        return
    with np.errstate(over="ignore"):
        idx = (h >> _HLL_SHIFT).astype(np.intp)
        w = (h << _HLL_LSHIFT) | _HLL_SENTINEL
    # Exact 64-bit bit_length via frexp on two exact 32-bit halves (no float rounding at 2^k)
    hi = (w >> _U32_SHIFT).astype(np.float64)
    lo = (w & _U32_MASK).astype(np.float64)
    _, e_hi = np.frexp(hi)
    _, e_lo = np.frexp(lo)
    bit_len = np.where(hi > 0.0, e_hi + 32, e_lo)
    rho = (65 - bit_len).astype(np.uint8)  # rho = leading_zeros + 1 = 64 - bit_len + 1
    np.maximum.at(regs, idx, rho)


def _ertl_sigma(x: float) -> float:
    if x == 1.0:
        return math.inf
    y, z = 1.0, x
    while True:
        x *= x
        z_old = z
        z += x * y
        y += y
        if z == z_old:
            return z


def _ertl_tau(x: float) -> float:
    if x == 0.0 or x == 1.0:
        return 0.0
    y, z = 1.0, 1.0 - x
    while True:
        x = math.sqrt(x)
        z_old = z
        y *= 0.5
        z -= (1.0 - x) ** 2 * y
        if z == z_old:
            return z / 3.0


def hll_estimate(regs: Union[np.ndarray, Sequence[int]]) -> int:
    """Ertl (2017) improved raw estimator: unbiased over the full range, no empirical tables."""
    r = np.asarray(regs, dtype=np.int64)
    if r.shape[0] != HLL_M:
        return 0
    np.clip(r, 0, HLL_Q + 1, out=r)
    C = np.bincount(r, minlength=HLL_Q + 2).astype(np.float64)
    m = float(HLL_M)
    if C[0] >= m:
        return 0
    z = m * _ertl_tau(1.0 - C[HLL_Q + 1] / m)
    for k in range(HLL_Q, 0, -1):
        z = 0.5 * (z + C[k])
    z += m * _ertl_sigma(C[0] / m)
    est = (m * m) / (2.0 * math.log(2.0)) / z
    return max(0, int(round(est)))


# ==============================================================================
# 3. SIGNED LOGARITHMIC DDSKETCH
# ==============================================================================
class DDSketchLog:
    __slots__ = ("bins", "neg_bins", "zero_count", "total_count")

    def __init__(self) -> None:
        self.bins: Dict[int, int] = {}
        self.neg_bins: Dict[int, int] = {}
        self.zero_count: int = 0
        self.total_count: int = 0

    def merge_dicts(self, pos_bins: Optional[Dict[int, int]], neg_bins: Optional[Dict[int, int]], zero_cnt: int) -> None:
        added = int(zero_cnt)
        self.zero_count += int(zero_cnt)
        for src, dst in ((pos_bins, self.bins), (neg_bins, self.neg_bins)):
            if not src:
                continue
            for k, v in src.items():
                iv = int(v)
                if iv > 0:
                    ik = int(k)
                    dst[ik] = dst.get(ik, 0) + iv
                    added += iv
        self.total_count += added

    def merge(self, other: "DDSketchLog") -> None:
        self.merge_dicts(other.bins, other.neg_bins, other.zero_count)

    def ordered(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ascending (representative_value, count) arrays across neg -> zero -> pos."""
        vals: List[float] = []
        cnts: List[int] = []
        for k in sorted(self.neg_bins.keys(), reverse=True):
            vals.append(-_bin_to_value(k))
            cnts.append(self.neg_bins[k])
        if self.zero_count > 0:
            vals.append(0.0)
            cnts.append(self.zero_count)
        for k in sorted(self.bins.keys()):
            vals.append(_bin_to_value(k))
            cnts.append(self.bins[k])
        return np.asarray(vals, dtype=np.float64), np.asarray(cnts, dtype=np.int64)

    @staticmethod
    def _quantiles_from(v: np.ndarray, c: np.ndarray, qs: Sequence[float]) -> List[float]:
        n = int(c.sum()) if c.size else 0
        if n <= 0:
            return [0.0] * len(qs)
        cum = np.cumsum(c)
        out: List[float] = []
        for q in qs:
            rank = max(0.0, min(1.0, float(q))) * (n - 1)
            i = int(np.searchsorted(cum, rank, side="right"))
            out.append(float(v[min(i, v.shape[0] - 1)]))
        return out

    @staticmethod
    def _es_lower_from(v: np.ndarray, c: np.ndarray, alpha: float) -> float:
        n = int(c.sum()) if c.size else 0
        if n <= 0:
            return 0.0
        cutoff = max(1.0, float(alpha) * n)
        cf = c.astype(np.float64)
        before = np.cumsum(cf) - cf
        take = np.clip(cutoff - before, 0.0, cf)
        return float(np.dot(take, v) / cutoff)

    def quantiles(self, qs: Sequence[float]) -> List[float]:
        v, c = self.ordered()
        return self._quantiles_from(v, c, qs)

    def quantile(self, q: float) -> float:
        return self.quantiles([q])[0]

    def expected_shortfall_lower(self, alpha: float = 0.05) -> float:
        v, c = self.ordered()
        return self._es_lower_from(v, c, alpha)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p": {str(k): int(v) for k, v in self.bins.items() if v > 0},
            "n": {str(k): int(v) for k, v in self.neg_bins.items() if v > 0},
            "z": int(self.zero_count),
            "t": int(self.total_count),
        }

    @classmethod
    def from_dict(cls, data: Union[str, Dict[str, Any], None]) -> "DDSketchLog":
        obj = cls()
        if not data:
            return obj
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (ValueError, TypeError):
                return obj
        if not isinstance(data, dict):
            return obj
        if "p" in data:
            obj.bins = {int(k): int(v) for k, v in data.get("p", {}).items() if int(v) > 0}
            neg_raw = data.get("n", {})
            if isinstance(neg_raw, dict):
                obj.neg_bins = {int(k): int(v) for k, v in neg_raw.items() if int(v) > 0}
            obj.zero_count = int(data.get("z", 0))
            obj.total_count = int(data.get("t", sum(obj.bins.values()) + sum(obj.neg_bins.values()) + obj.zero_count))
        elif "b" in data:
            for k_str, v in data.get("b", {}).items():
                iv = int(v)
                if iv > 0:
                    ik = int(k_str)
                    exp_k = ik - 8192 if ik > 2048 else ik
                    obj.bins[exp_k] = obj.bins.get(exp_k, 0) + iv
            obj.zero_count = int(data.get("z", 0))
            obj.total_count = int(data.get("t", sum(obj.bins.values()) + obj.zero_count))
        return obj


# ==============================================================================
# 4. MERGEABLE STREAMING SKETCH STATE
# ==============================================================================
_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


class MergeableSketchState:
    __slots__ = (
        "count", "mean", "m2", "min_val", "max_val",
        "pos_sum", "neg_sum", "downside_m2", "pos_count", "neg_count", "zero_count",
        "ddsketch", "hll_registers",
    )

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")
        self.pos_sum: float = 0.0
        self.neg_sum: float = 0.0
        self.downside_m2: float = 0.0
        self.pos_count: int = 0
        self.neg_count: int = 0
        self.zero_count: int = 0
        self.ddsketch: DDSketchLog = DDSketchLog()
        self.hll_registers: np.ndarray = np.zeros(HLL_M, dtype=np.uint8)

    def merge_batch_stats(
        self, b_n: int, b_min: float, b_max: float, b_mean: float, b_m2: float,
        b_bins: Optional[Dict[int, int]] = None, b_neg_bins: Optional[Dict[int, int]] = None,
        b_zero_count: int = 0, b_pos_sum: float = 0.0, b_neg_sum: float = 0.0,
        b_downside_m2: float = 0.0, b_pos_count: int = 0, b_neg_count: int = 0,
    ) -> None:
        if b_n <= 0:
            return
        b_n = int(b_n)
        if self.count == 0:
            self.count, self.mean, self.m2 = b_n, float(b_mean), float(b_m2)
            self.min_val, self.max_val = float(b_min), float(b_max)
        else:
            new_cnt = self.count + b_n
            delta = float(b_mean) - self.mean
            self.mean += delta * (b_n / new_cnt)
            self.m2 += float(b_m2) + (delta * delta) * (self.count * (b_n / new_cnt))
            self.min_val = min(self.min_val, float(b_min))
            self.max_val = max(self.max_val, float(b_max))
            self.count = new_cnt
        self.pos_sum += float(b_pos_sum)
        self.neg_sum += float(b_neg_sum)
        self.downside_m2 += float(b_downside_m2)
        self.pos_count += int(b_pos_count)
        self.neg_count += int(b_neg_count)
        self.zero_count += int(b_zero_count)
        self.ddsketch.merge_dicts(b_bins, b_neg_bins, b_zero_count)

    def merge(self, other: "MergeableSketchState") -> None:
        if other.count > 0:
            self.merge_batch_stats(
                b_n=other.count, b_min=other.min_val, b_max=other.max_val,
                b_mean=other.mean, b_m2=other.m2,
                b_bins=other.ddsketch.bins, b_neg_bins=other.ddsketch.neg_bins,
                b_zero_count=other.ddsketch.zero_count,
                b_pos_sum=other.pos_sum, b_neg_sum=other.neg_sum,
                b_downside_m2=other.downside_m2,
                b_pos_count=other.pos_count, b_neg_count=other.neg_count,
            )
        o = np.asarray(other.hll_registers, dtype=np.uint8)
        if o.shape[0] == HLL_M:
            if not isinstance(self.hll_registers, np.ndarray):
                self.hll_registers = np.asarray(self.hll_registers, dtype=np.uint8)
            np.maximum(self.hll_registers, o, out=self.hll_registers)

    def estimate_cardinality(self) -> int:
        return hll_estimate(self.hll_registers)

    def finalize_metrics(self) -> Dict[str, Any]:
        card = self.estimate_cardinality()
        if self.count <= 0:
            z = {k: 0.0 for k in (
                "mean", "stddev", "variance", "min", "max", "p01", "p05", "p25", "p50", "p75",
                "p95", "p99", "iqr", "sharpe_ratio", "sortino_ratio", "profit_factor", "win_rate_pct",
                "var_95", "var_99", "cvar_95", "cvar_99", "pos_sum", "neg_sum")}
            z.update({"count": 0, "cardinality": card, "unique_categories": card, "financial_status": "IDLE"})
            return z

        variance = max(0.0, self.m2 / (self.count - 1)) if self.count > 1 else 0.0
        stddev = math.sqrt(variance)
        safe_min = float(self.min_val if math.isfinite(self.min_val) else 0.0)
        safe_max = float(self.max_val if math.isfinite(self.max_val) else 0.0)

        def _clamp(val: float) -> float:
            return max(safe_min, min(safe_max, val))

        v, c = self.ddsketch.ordered()
        p01, p05, p25, p50, p75, p95, p99 = (_clamp(x) for x in DDSketchLog._quantiles_from(v, c, _QUANTILES))
        iqr = max(0.0, p75 - p25)
        cvar_95 = _clamp(DDSketchLog._es_lower_from(v, c, 0.05))
        cvar_99 = _clamp(DDSketchLog._es_lower_from(v, c, 0.01))

        sharpe = (self.mean / stddev) if stddev > ZERO_EPSILON else 0.0
        downside_std = math.sqrt(max(0.0, self.downside_m2 / self.count))
        sortino = (self.mean / downside_std) if downside_std > ZERO_EPSILON else sharpe
        if self.neg_sum > ZERO_EPSILON:
            profit_factor = self.pos_sum / self.neg_sum
        else:
            profit_factor = 999.0 if self.pos_sum > ZERO_EPSILON else 0.0
        win_rate = (self.pos_count / self.count) * 100.0

        if self.neg_count > 0 and profit_factor < 1.0:
            fin_status = "CRITICAL_DRAWDOWN"
        elif stddev > abs(self.mean) * 2.5 and self.count > 100:
            fin_status = "HIGH_VOLATILITY"
        else:
            fin_status = "HEALTHY_OPTIMAL"

        r6 = lambda x: round(float(x), 6)  # noqa: E731
        return {
            "count": int(self.count), "mean": r6(self.mean), "stddev": r6(stddev), "variance": r6(variance),
            "min": r6(safe_min), "max": r6(safe_max),
            "p01": r6(p01), "p05": r6(p05), "p25": r6(p25), "p50": r6(p50),
            "p75": r6(p75), "p95": r6(p95), "p99": r6(p99), "iqr": r6(iqr),
            "cardinality": int(card), "unique_categories": int(card),
            "sharpe_ratio": round(float(sharpe), 4), "sortino_ratio": round(float(sortino), 4),
            "profit_factor": round(float(profit_factor), 4), "win_rate_pct": round(float(win_rate), 2),
            "var_95": r6(p05), "var_99": r6(p01), "cvar_95": r6(cvar_95), "cvar_99": r6(cvar_99),
            "pos_sum": round(float(self.pos_sum), 4), "neg_sum": round(float(self.neg_sum), 4),
            "financial_status": fin_status,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": int(self.count), "mean": float(self.mean), "m2": float(self.m2),
            "min_val": None if not math.isfinite(self.min_val) else float(self.min_val),
            "max_val": None if not math.isfinite(self.max_val) else float(self.max_val),
            "pos_sum": float(self.pos_sum), "neg_sum": float(self.neg_sum),
            "downside_m2": float(self.downside_m2),
            "pos_count": int(self.pos_count), "neg_count": int(self.neg_count),
            "zero_count": int(self.zero_count),
            "ddsketch": self.ddsketch.to_dict(),
            "hll": np.asarray(self.hll_registers, dtype=np.uint8).tolist(),
            "hll_v": HLL_VERSION,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MergeableSketchState":
        st = cls()
        if not d or not isinstance(d, dict):
            return st
        st.count = int(d.get("count", 0))
        st.mean = float(d.get("mean", 0.0))
        st.m2 = float(d.get("m2", 0.0))
        st.min_val = float(d["min_val"]) if d.get("min_val") is not None else float("inf")
        st.max_val = float(d["max_val"]) if d.get("max_val") is not None else float("-inf")
        st.pos_sum = float(d.get("pos_sum", 0.0))
        st.neg_sum = float(d.get("neg_sum", 0.0))
        st.downside_m2 = float(d.get("downside_m2", 0.0))
        st.pos_count = int(d.get("pos_count", 0))
        st.neg_count = int(d.get("neg_count", 0))
        st.zero_count = int(d.get("zero_count", 0))
        st.ddsketch = DDSketchLog.from_dict(d.get("ddsketch"))
        hll_list = d.get("hll")
        if hll_list and len(hll_list) == HLL_M:
            regs = np.clip(np.asarray(hll_list, dtype=np.int64), 0, HLL_Q + 2)
            if int(d.get("hll_v", 1)) < HLL_VERSION:
                # Legacy v1 registers carried a deterministic +1 rho offset -> undo it.
                regs = np.where(regs > 0, regs - 1, 0)
            st.hll_registers = np.clip(regs, 0, HLL_Q + 1).astype(np.uint8)
        return st


# ==============================================================================
# 5. ZERO-OVERHEAD HARDWARE PROFILER
# ==============================================================================
class ZeroOverheadProfiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_ts: float = 0.0
        self._cached_hw: Dict[str, Any] = {}
        self._proc = psutil.Process(os.getpid())
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def sample(self) -> Dict[str, Any]:
        now = time.perf_counter()
        with self._lock:
            if (now - self._last_ts) < 0.40 and self._cached_hw:
                return self._cached_hw
            try:
                vm = psutil.virtual_memory()
                sw = psutil.swap_memory()
                cpu_pct = psutil.cpu_percent(interval=None)
                proc_mem_mb = self._proc.memory_info().rss / 1048576.0
            except Exception:
                vm, sw, cpu_pct, proc_mem_mb = None, None, 0.0, 0.0
            try:
                pool = pa.default_memory_pool()
                arrow_mb = pool.bytes_allocated() / 1048576.0
                arrow_backend = pool.backend_name
            except Exception:
                arrow_mb, arrow_backend = 0.0, "unknown"
            payload = {
                "cpu_pct": round(float(cpu_pct), 1),
                "cpu": round(float(cpu_pct), 1),
                "ram_used_gb": round(vm.used / 1073741824.0, 2) if vm else 0.0,
                "ram_total_gb": round(vm.total / 1073741824.0, 2) if vm else 16.0,
                "ram_avail_gb": round(vm.available / 1073741824.0, 2) if vm else 16.0,
                "ram_pct": round(vm.percent, 1) if vm else 0.0,
                "ram_mb": round(proc_mem_mb, 1),
                "swap_used_mb": round(sw.used / 1048576.0, 1) if sw else 0.0,
                "swap_pct": round(sw.percent, 1) if sw else 0.0,
                "arrow_pool_mb": round(arrow_mb, 1),
                "arrow_backend": arrow_backend,
            }
            self._last_ts = now
            self._cached_hw = payload
            return payload


# ==============================================================================
# 6. RAM-CACHED ACID SQLITE WAL LEDGER
# ==============================================================================
class ManifestLedger:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._version: int = 1
        self._ns_sketches: Dict[str, MergeableSketchState] = {}
        self._ns_cache: Dict[str, Dict[str, Any]] = {}
        self._recent_cache: List[Dict[str, Any]] = []
        self._init_db_and_warm_cache()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def _init_db_and_warm_cache(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS file_ledger (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL, parquet_file TEXT, namespace TEXT NOT NULL,
                        rows INTEGER NOT NULL, raw_mb REAL NOT NULL, parquet_mb REAL NOT NULL,
                        compression_ratio REAL NOT NULL, elapsed_sec REAL NOT NULL,
                        mb_per_sec REAL NOT NULL, rows_per_sec REAL NOT NULL,
                        num_col TEXT, cat_col TEXT, metrics_json TEXT NOT NULL,
                        sketch_json TEXT NOT NULL, ingested_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_ledger_ns ON file_ledger(namespace);
                    CREATE INDEX IF NOT EXISTS idx_ledger_time ON file_ledger(ingested_at DESC);
                    """
                )
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(file_ledger)").fetchall()}
                if "parquet_file" not in cols:
                    conn.execute("ALTER TABLE file_ledger ADD COLUMN parquet_file TEXT DEFAULT '';")
                conn.commit()
                all_rows = conn.execute(
                    """
                    SELECT filename, parquet_file, namespace, rows, raw_mb, parquet_mb,
                           compression_ratio, elapsed_sec, mb_per_sec, rows_per_sec,
                           num_col, cat_col, metrics_json, sketch_json, ingested_at
                    FROM file_ledger ORDER BY ingested_at ASC
                    """
                ).fetchall()
            finally:
                conn.close()

            self._ns_sketches.clear()
            self._ns_cache.clear()
            self._recent_cache.clear()
            for r in all_rows:
                s_json = r["sketch_json"]
                try:
                    sketch = MergeableSketchState.from_dict(json.loads(s_json)) if s_json else MergeableSketchState()
                except (ValueError, TypeError):
                    sketch = MergeableSketchState()
                # Recompute file metrics from the sketch so legacy HLL/DD bugs are healed on load
                file_metrics = sketch.finalize_metrics()
                self._merge_into_ram_cache(
                    namespace=str(r["namespace"] or "schema_ns_01"),
                    filename=str(r["filename"]),
                    parquet_file=str(r["parquet_file"] or ""),
                    rows=int(r["rows"] or 0),
                    raw_mb=float(r["raw_mb"] or 0.0),
                    comp_mb=float(r["parquet_mb"] or 0.0),
                    elapsed_s=float(r["elapsed_sec"] or 0.0),
                    num_col=str(r["num_col"] or ""),
                    cat_col=str(r["cat_col"] or ""),
                    sketch_state=sketch,
                    file_metrics=file_metrics,
                    ingested_at=float(r["ingested_at"] or time.time()),
                )

    def _merge_into_ram_cache(
        self, namespace: str, filename: str, parquet_file: str, rows: int, raw_mb: float,
        comp_mb: float, elapsed_s: float, num_col: str, cat_col: str,
        sketch_state: MergeableSketchState, file_metrics: Dict[str, Any], ingested_at: float,
    ) -> None:
        ns_sketch = self._ns_sketches.setdefault(namespace, MergeableSketchState())
        ns_sketch.merge(sketch_state)
        ns_kpis = ns_sketch.finalize_metrics()

        prev = self._ns_cache.get(namespace, {
            "namespace": namespace, "files_count": 0, "parquet_files": 0, "total_rows": 0,
            "raw_mb": 0.0, "comp_mb": 0.0, "total_elapsed_s": 0.0,
            "num_col": num_col, "cat_col": cat_col,
        })
        tot_files = int(prev["files_count"]) + 1
        pq_files = int(prev["parquet_files"]) + 1
        tot_rows = int(prev["total_rows"]) + int(rows)
        tot_raw = float(prev["raw_mb"]) + float(raw_mb)
        tot_comp = float(prev["comp_mb"]) + float(comp_mb)
        tot_el = float(prev["total_elapsed_s"]) + float(elapsed_s)

        self._ns_cache[namespace] = {
            "namespace": namespace, "files_count": tot_files, "parquet_files": pq_files,
            "total_rows": tot_rows, "raw_mb": round(tot_raw, 2), "comp_mb": round(tot_comp, 2),
            "compression_ratio": round(tot_raw / max(0.01, tot_comp), 2),
            "avg_speed_mbps": round(tot_raw / max(0.001, tot_el), 2),
            "total_elapsed_s": round(tot_el, 2),
            "num_col": num_col or prev.get("num_col", ""),
            "cat_col": cat_col or prev.get("cat_col", ""),
            "kpis": ns_kpis, "metrics": ns_kpis,
            "mean": ns_kpis.get("mean", 0.0), "stddev": ns_kpis.get("stddev", 0.0),
            "min": ns_kpis.get("min", 0.0), "max": ns_kpis.get("max", 0.0),
            "p50": ns_kpis.get("p50", 0.0), "p95": ns_kpis.get("p95", 0.0), "p99": ns_kpis.get("p99", 0.0),
            "cardinality": ns_kpis.get("cardinality", 0),
            "unique_categories": ns_kpis.get("cardinality", 0),
        }

        spd_file = round(raw_mb / max(0.001, elapsed_s), 2)
        entry = {
            "filename": filename, "file": filename, "parquet_file": parquet_file, "namespace": namespace,
            "rows": int(rows), "raw_mb": round(float(raw_mb), 2), "comp_mb": round(float(comp_mb), 2),
            "parquet_mb": round(float(comp_mb), 2),
            "compression_ratio": round(raw_mb / max(0.01, comp_mb), 2),
            "elapsed_s": round(float(elapsed_s), 2), "elapsed_sec": round(float(elapsed_s), 2),
            "speed_mbps": spd_file, "mb_per_sec": spd_file,
            "rows_per_sec": round(rows / max(0.001, elapsed_s), 0),
            "num_col": num_col, "cat_col": cat_col,
            "kpis": file_metrics, "metrics": file_metrics, "ingested_at": ingested_at,
        }
        self._recent_cache.insert(0, entry)
        if len(self._recent_cache) > 200:
            del self._recent_cache[200:]

    def commit_file_ingestion(
        self, namespace: str, filename: str, parquet_file: str, rows: int, raw_mb: float,
        comp_mb: float, elapsed_s: float, num_col: str, cat_col: str,
        sketch_state: MergeableSketchState,
    ) -> Dict[str, Any]:
        file_metrics = sketch_state.finalize_metrics()
        now_ts = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO file_ledger (
                        filename, parquet_file, namespace, rows, raw_mb, parquet_mb,
                        compression_ratio, elapsed_sec, mb_per_sec, rows_per_sec,
                        num_col, cat_col, metrics_json, sketch_json, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        filename, parquet_file or "", namespace, int(rows), float(raw_mb), float(comp_mb),
                        float(round(raw_mb / max(0.01, comp_mb), 2)), round(float(elapsed_s), 2),
                        float(round(raw_mb / max(0.001, elapsed_s), 2)),
                        float(round(rows / max(0.001, elapsed_s), 0)),
                        num_col or "", cat_col or "",
                        json.dumps(file_metrics), json.dumps(sketch_state.to_dict()), now_ts,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
            self._merge_into_ram_cache(
                namespace=namespace, filename=filename, parquet_file=parquet_file, rows=rows,
                raw_mb=raw_mb, comp_mb=comp_mb, elapsed_s=elapsed_s, num_col=num_col, cat_col=cat_col,
                sketch_state=sketch_state, file_metrics=file_metrics, ingested_at=now_ts,
            )
            self._version += 1
            return self._recent_cache[0]

    def record_ingestion(
        self, filename: str, namespace: str, rows: int, raw_mb: float, parquet_mb: float,
        elapsed_sec: float, num_col: str, cat_col: str, sketch_state: MergeableSketchState,
        parquet_file: str = "",
    ) -> Dict[str, Any]:
        return self.commit_file_ingestion(
            namespace=namespace, filename=filename, parquet_file=parquet_file, rows=rows,
            raw_mb=raw_mb, comp_mb=parquet_mb, elapsed_s=elapsed_sec,
            num_col=num_col, cat_col=cat_col, sketch_state=sketch_state,
        )

    def record_compaction(self, namespace: str, old_count: int, new_count: int = 1) -> None:
        with self._lock:
            if namespace in self._ns_cache:
                self._ns_cache[namespace]["parquet_files"] = int(new_count)
            self._version += 1

    def get_namespaces(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._ns_cache)

    def get_recent_files(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._recent_cache[: int(limit)])

    def get_file_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.get_recent_files(limit=limit)

    def get_global_kpis(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if namespace and namespace not in ("all", "default", "*") and namespace in self._ns_cache:
                info = self._ns_cache[namespace]
                return {
                    "files_processed": info["files_count"], "total_rows": info["total_rows"],
                    "total_raw_mb": info["raw_mb"], "total_parquet_mb": info["comp_mb"],
                    "compression_ratio": info["compression_ratio"], "avg_mb_per_sec": info["avg_speed_mbps"],
                    "num_col": info["num_col"], "cat_col": info["cat_col"],
                    "metrics": info["kpis"], "kpis": info["kpis"],
                }
            if not self._ns_cache:
                empty = MergeableSketchState().finalize_metrics()
                return {
                    "files_processed": 0, "total_rows": 0, "total_raw_mb": 0.0, "total_parquet_mb": 0.0,
                    "compression_ratio": 0.0, "avg_mb_per_sec": 0.0, "num_col": "value",
                    "cat_col": "category", "metrics": empty, "kpis": empty,
                }
            merged = MergeableSketchState()
            tot_files = tot_rows = 0
            tot_raw = tot_comp = tot_el = 0.0
            last_num, last_cat = "value", "category"
            for ns_name, info in self._ns_cache.items():
                tot_files += int(info["files_count"])
                tot_rows += int(info["total_rows"])
                tot_raw += float(info["raw_mb"])
                tot_comp += float(info["comp_mb"])
                tot_el += float(info["total_elapsed_s"])
                last_num = info.get("num_col") or last_num
                last_cat = info.get("cat_col") or last_cat
                if ns_name in self._ns_sketches:
                    merged.merge(self._ns_sketches[ns_name])
            metrics = merged.finalize_metrics()
            return {
                "files_processed": tot_files, "total_rows": tot_rows,
                "total_raw_mb": round(tot_raw, 2), "total_parquet_mb": round(tot_comp, 2),
                "compression_ratio": round(tot_raw / max(0.01, tot_comp), 2),
                "avg_mb_per_sec": round(tot_raw / max(0.001, tot_el), 2),
                "num_col": last_num, "cat_col": last_cat, "metrics": metrics, "kpis": metrics,
            }

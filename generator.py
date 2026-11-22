# generator.py
# TITAN STRESS DATA GENERATOR — v6.2.0-SOTA (50GB EDITION + KERNEL FSYNC)

import argparse
import os
import random
import shutil
import sys
import time
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.csv as pv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INBOX_DIR = os.path.join(BASE_DIR, "inbox")
FAILED_INBOX_DIR = os.path.join(BASE_DIR, "inbox_failed")
os.makedirs(INBOX_DIR, exist_ok=True)
os.makedirs(FAILED_INBOX_DIR, exist_ok=True)

# سایز هدف: ۵۰ گیگابایت (۵۱,۲۰۰ مگابایت)
DEFAULT_TARGET_MB = 51200


def _build_seed_buffers(
    schema_type: str,
    num_buffers: int = 16,
    rows_per_buf: int = 100_000,
) -> Tuple[bytes, List[Tuple[bytes, int]]]:
    """
    ساخت بافرهای باینری CSV در سطح C++ PyArrow و NumPy با توزیع آماری واقعی (Gaussian).
    هر بافر شامل ۱۰۰ هزار ردیف (~4.5MB) است تا بدون فشار به CPU، دیسک را با حداکثر سرعت ترتیبی پر کند.
    """
    buffers: List[Tuple[bytes, int]] = []
    write_opts = pv.WriteOptions(include_header=False)
    header_bytes = b""

    rng = np.random.default_rng(seed=2026)
    lines_pool = np.array([f"LINE_{i:02d}" for i in range(1, 51)], dtype=object)
    cities_pool = np.array(["Tehran", "Shiraz", "Tabriz", "Isfahan", "Mashhad"], dtype=object)

    for b_idx in range(num_buffers):
        # تولید برداریِ تایم‌استمپ‌ها برای ۶۰ ثانیه و تکرار فوق‌سریع آن‌ها
        sec_templates = np.array(
            [f"2026-11-05 06:{b_idx:02d}:{s:02d}" for s in range(60)],
            dtype=object,
        )
        timestamps = np.resize(sec_templates, rows_per_buf)

        if schema_type == "factory":
            sensor_ids = rng.integers(1000, 1_000_000, size=rows_per_buf, dtype=np.int64)
            machine_lines = lines_pool[rng.integers(0, len(lines_pool), size=rows_per_buf)]
            temperatures = np.round(rng.normal(75.0, 12.5, size=rows_per_buf), 4)

            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array(sensor_ids, type=pa.int64()),
                    pa.array(machine_lines, type=pa.string()),
                    pa.array(temperatures, type=pa.float64()),
                    pa.array(timestamps, type=pa.string()),
                ],
                names=["sensor_id", "machine_line", "temperature", "timestamp"],
            )
        else:
            order_ids = rng.integers(100_000, 10_000_000, size=rows_per_buf, dtype=np.int64)
            cities = cities_pool[rng.integers(0, len(cities_pool), size=rows_per_buf)]
            amounts = np.round(rng.normal(250.0, 45.0, size=rows_per_buf), 2)

            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array(order_ids, type=pa.int64()),
                    pa.array(cities, type=pa.string()),
                    pa.array(amounts, type=pa.float64()),
                    pa.array(timestamps, type=pa.string()),
                ],
                names=["order_id", "city", "amount", "timestamp"],
            )

        if b_idx == 0:
            h_sink = pa.BufferOutputStream()
            pv.write_csv(batch.slice(0, 1), h_sink, write_options=pv.WriteOptions(include_header=True))
            full_first = h_sink.getvalue().to_pybytes()
            header_bytes = full_first.split(b"\n", 1)[0] + b"\n"

        sink = pa.BufferOutputStream()
        pv.write_csv(batch, sink, write_options=write_opts)
        buffers.append((sink.getvalue().to_pybytes(), rows_per_buf))

    return header_bytes, buffers


def generate_csv_file(
    filename: str,
    target_mb: float = DEFAULT_TARGET_MB,
    schema_type: str = "factory",
) -> dict:
    os.makedirs(INBOX_DIR, exist_ok=True)
    final_path = os.path.join(INBOX_DIR, filename)
    tmp_path = os.path.join(INBOX_DIR, f".{filename}.tmp")

    # بررسی فضای خالی دیسک قبل از نوشتن فایل ۵۰ گیگابایتی
    free_bytes = shutil.disk_usage(os.path.abspath(INBOX_DIR)).free
    free_gb = free_bytes / (1024 ** 3)
    needed_gb = (target_mb / 1024.0) * 1.22  # حجم CSV + فضای موقت Parquet (~22%)
    print(f"[Disk Check] Free space on drive: {free_gb:,.1f} GB (Recommended required: ~{needed_gb:,.1f} GB)")
    if free_gb < needed_gb:
        print(
            f"[WARNING] Free disk space ({free_gb:,.1f} GB) is tight for "
            f"{target_mb / 1024:.0f} GB CSV + Parquet output!"
        )

    target_bytes = int(target_mb * 1024 * 1024)
    t0 = time.perf_counter()

    rows_per_chunk = 100_000 if target_mb >= 1024 else 50_000
    print(f"[Generator] Building vectorized C++ seed buffers for {filename} ({target_mb:,.0f} MB)...")
    header_bytes, buffers = _build_seed_buffers(
        schema_type=schema_type,
        num_buffers=16,
        rows_per_buf=rows_per_chunk,
    )

    written_bytes = 0
    total_rows = 0
    last_print = time.perf_counter()

    # بافر ۳۲ مگابایتی برای به حداکثر رساندن توان نوشتن ترتیبی در فایل ۵۰ گیگابایتی
    with open(tmp_path, "wb", buffering=32 * 1024 * 1024) as f:
        f.write(header_bytes)
        written_bytes += len(header_bytes)

        while written_bytes < target_bytes:
            buf_bytes, r_cnt = random.choice(buffers)
            f.write(buf_bytes)
            written_bytes += len(buf_bytes)
            total_rows += r_cnt

            now = time.perf_counter()
            if now - last_print >= 1.0:
                pct = min(100.0, written_bytes / target_bytes * 100.0)
                mb_done = written_bytes / 1048576
                elapsed_so_far = max(0.001, now - t0)
                speed = mb_done / elapsed_so_far
                rem_mb = max(0.0, target_mb - mb_done)
                eta_s = int(rem_mb / max(1.0, speed))
                sys.stdout.write(
                    f"\r[Generator] {mb_done:,.0f}/{target_mb:,.0f} MB ({pct:.1f}%) | "
                    f"Rows: {total_rows:,} | Write: {speed:,.0f} MB/s | ETA: {eta_s}s   "
                )
                sys.stdout.flush()
                last_print = now

        f.flush()
        # تضمین تخلیه کامل صفحات کثیف کش ویندوز روی دیسک قبل از تحویل فایل به موتور
        os.fsync(f.fileno())
        actual_bytes = f.tell()

    # تغییر نام اتمیک بعد از تکمیل و تثبیت کامل ۵۰ گیگابایت روی دیسک
    os.replace(tmp_path, final_path)
    elapsed = time.perf_counter() - t0
    actual_mb = actual_bytes / 1048576

    print(
        f"\n[Generator] DONE! Created {filename}: {actual_mb:,.1f} MB "
        f"({total_rows:,} rows) in {elapsed:.2f}s ({actual_mb / max(0.01, elapsed):,.1f} MB/s)"
    )
    return {
        "file": filename,
        "mb": round(actual_mb, 2),
        "rows": total_rows,
        "sec": round(elapsed, 2),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic CSV stress file inside inbox/."
    )
    parser.add_argument(
        "--mb",
        type=float,
        default=DEFAULT_TARGET_MB,
        help=f"Target size in MiB (default: {DEFAULT_TARGET_MB:,}).",
    )
    parser.add_argument(
        "--schema",
        choices=("factory", "orders"),
        default="factory",
        help="CSV schema to generate (default: factory).",
    )
    parser.add_argument(
        "--name",
        default="1_titan_50gb_stress.csv",
        help="Output filename inside inbox/ (default: 1_titan_50gb_stress.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # تولید فایل تست فشار با اندازه و اسکیمای دلخواه؛ فایل‌های دیگر inbox دست‌نخورده می‌مانند.
    generate_csv_file(args.name, target_mb=args.mb, schema_type=args.schema)
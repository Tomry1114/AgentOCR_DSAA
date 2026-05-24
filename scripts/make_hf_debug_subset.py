#!/usr/bin/env python3
import argparse
import os

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Create a tiny parquet subset while preserving HF image metadata.")
    parser.add_argument("--src", required=True, help="Source parquet path")
    parser.add_argument("--dst", required=True, help="Destination parquet path")
    parser.add_argument("--rows", type=int, required=True, help="Number of rows to keep")
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.src)
    full_table = parquet.read()
    keep_rows = min(args.rows, full_table.num_rows)
    subset = full_table.slice(0, keep_rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    pq.write_table(subset, args.dst)

    verify = pq.ParquetFile(args.dst)
    metadata = verify.schema_arrow.metadata or {}
    first_row = verify.read().slice(0, 1).to_pylist()[0] if keep_rows > 0 else {}
    first_image = None
    if first_row.get("images"):
        first_image = first_row["images"][0]

    print(
        {
            "src": args.src,
            "dst": args.dst,
            "rows": keep_rows,
            "has_huggingface_metadata": b"huggingface" in metadata,
            "schema": str(verify.schema_arrow),
            "first_image_type": type(first_image).__name__ if first_image is not None else None,
        }
    )


if __name__ == "__main__":
    main()

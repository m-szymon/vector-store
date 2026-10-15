#!/usr/bin/env python3
"""Write the embedding files hybrid.rn reads next to a prepared dataset.

hybrid.rn pairs each TSV with an fbin of the same name holding one vector per
TSV record, in file order. For a directory laid out by prepare_dataset.py
this writes:

  documents.fbin           for documents.tsv, if present
  shards/documents_NNN.fbin  for each shard, if the dataset is sharded
  queries_<set>.fbin       for each queries_<set>.tsv

The documents are one ordered sequence, documents.tsv or its shards in order,
and get one vector stream: shard N's vectors are the matching slice of
documents.fbin, so loading the shards one by one indexes the same vectors as
loading documents.tsv. Each query file has a stream of its own.

Records are counted the way the workload reads them - lines end at \\n, blank
lines are skipped and any other line must hold a tab with an id before it -
so a malformed TSV fails here, naming the line, rather than at the workload's
prepare. Every file is counted before any is written, and the outputs are
written under temporary names and renamed only when all are complete.

The vectors are random: unit-length, uniform on the sphere and drawn from
--seed. They give the vector index a realistic size and dimension for latency
and throughput, but carry no meaning: the vector leg's ranking, and so the
fused one, is noise. Random high-dimensional vectors are also a harder case
for an ANN index than real embeddings, so read ANN latency from them as
pessimistic. If the directory has a manifest.json, the dimension, seed and
files are recorded in it under "embeddings".

Needs numpy: drawing 10M x 768 vectors takes about a minute with it and half
an hour in pure Python. Run it after prepare_dataset.py, and again whenever
that rewrites the dataset.

  gen_embeddings.py <dataset_dir> --dimension 768
"""
import argparse
import json
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

# The characters Rust's str::trim strips (Unicode White_Space), which the
# workload's TSV reader applies. Python's str.strip() also strips the
# \x1c-\x1f separators, which would make a line of them blank here but not
# in the workload.
WHITESPACE = (
    "\t\n\x0b\x0c\r \x85\xa0       "
    "         　"
)

# Vectors drawn at a time, about 64 MiB of float32 whatever the dimension.
CHUNK_BYTES = 64 << 20


def count_records(path):
    """Count the records hybrid.rn reads from a TSV, failing on a line it
    would reject. Lines end at \\n only, as in the workload's reader."""
    count = 0
    with path.open("rb") as tsv:
        for line_no, raw in enumerate(tsv, start=1):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                sys.exit(f"{path}:{line_no}: not UTF-8 ({e.reason})")
            if line.strip(WHITESPACE) == "":
                continue
            record_id, tab, _ = line.partition("\t")
            if not tab:
                sys.exit(f'{path}:{line_no}: expected "id<TAB>text", found no tab')
            if record_id.strip(WHITESPACE) == "":
                sys.exit(f"{path}:{line_no}: no id before the tab")
            count += 1
    return count


def partial_path(path):
    return path.with_name(path.name + ".partial")


def write_stream(targets, dimension, seed, stream):
    """Write one random stream across targets, a list of (fbin path, record
    count), in order: each file takes the next count vectors. The stream is
    drawn in chunks sized only by the total, so any split of the same total
    into files writes the same vectors. Files go to their partial names."""
    rng = np.random.default_rng([seed, zlib.crc32(stream.encode())])
    total = sum(count for _, count in targets)
    chunk = max(1, CHUNK_BYTES // (dimension * 4))
    drawn = 0
    pending = np.empty((0, dimension), dtype=np.float32)
    for path, count in targets:
        with partial_path(path).open("wb") as out:
            out.write(struct.pack("<II", count, dimension))
            needed = count
            while needed:
                if len(pending) == 0:
                    rows = min(chunk, total - drawn)
                    pending = rng.standard_normal((rows, dimension), dtype=np.float32)
                    pending /= np.linalg.norm(pending, axis=1, keepdims=True)
                    drawn += rows
                take = min(needed, len(pending))
                pending[:take].astype("<f4", copy=False).tofile(out)
                pending = pending[take:]
                needed -= take


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset_dir", type=Path, help="prepared dataset directory")
    parser.add_argument(
        "--dimension",
        type=int,
        required=True,
        help="vector dimension; pass the same -P dimension to hybrid.rn",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    args = parser.parse_args()
    if args.dimension < 1:
        parser.error("--dimension must be at least 1")
    root = args.dataset_dir

    full = root / "documents.tsv"
    full = full if full.is_file() else None
    shards = sorted((root / "shards").glob("documents_*.tsv"))
    queries = sorted(root.glob("queries_*.tsv"))
    if full is None and not shards:
        parser.error(f"{root}: no documents.tsv and no shards/documents_*.tsv")
    if not queries:
        # Enough to load; hybrid.rn's search phase will need a query set.
        print(f"{root}: no queries_*.tsv, writing document embeddings only")

    # Count every file before writing any.
    counts = {tsv: count_records(tsv) for tsv in [*([full] if full else []), *shards, *queries]}
    if full and shards and sum(counts[s] for s in shards) != counts[full]:
        sys.exit(
            f"{root}: the shards hold {sum(counts[s] for s in shards)} documents but "
            f"documents.tsv holds {counts[full]}; re-run prepare_dataset.py"
        )

    streams = []
    if full:
        streams.append(("documents", [full]))
    if shards:
        streams.append(("documents", shards))
    streams += [(tsv.name, [tsv]) for tsv in queries]
    for stream, tsvs in streams:
        targets = [(tsv.with_suffix(".fbin"), counts[tsv]) for tsv in tsvs]
        write_stream(targets, args.dimension, args.seed, stream)

    outputs = [tsv.with_suffix(".fbin") for _, tsvs in streams for tsv in tsvs]
    for fbin in outputs:
        os.replace(partial_path(fbin), fbin)
        print(f"{fbin}: {counts[fbin.with_suffix('.tsv')]} x {args.dimension}")

    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest["embeddings"] = {
            "generator": "random",
            "dimension": args.dimension,
            "seed": args.seed,
            "files": {
                str(fbin.relative_to(root)): counts[fbin.with_suffix(".tsv")] for fbin in outputs
            },
        }
        partial = partial_path(manifest_path)
        partial.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(partial, manifest_path)
        print(f"{manifest_path}: recorded under \"embeddings\"")


if __name__ == "__main__":
    main()

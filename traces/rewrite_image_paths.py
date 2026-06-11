#!/usr/bin/env python3
"""Remap the image-path prefix inside the shipped trace JSONs.

The trace datasets in this repo store absolute TU Delft cluster paths in each
record's "images" list, e.g.

    /bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images/...

If you work on a different machine (or moved the downloaded image collections),
rewrite that prefix in place:

    python rewrite_image_paths.py \
        --old-prefix /bulk/aacudad/reasoning_traces/reasoning_traces_gen/data \
        --new-prefix /my/data \
        anomalythink_6k/*.json anomalythink_15k/**/*.json

A `.bak` copy of every modified file is written next to it (use --no-backup to
skip). Use --check to report how many image paths exist on disk after the
rewrite (dry verification, no extra changes).
"""
import argparse
import json
import os
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="trace JSON files to rewrite")
    ap.add_argument("--old-prefix", required=True)
    ap.add_argument("--new-prefix", required=True)
    ap.add_argument("--no-backup", action="store_true", help="do not keep a .bak copy")
    ap.add_argument("--check", action="store_true",
                    help="after rewriting, count which image paths exist on disk")
    args = ap.parse_args()

    total_rewritten = 0
    for path in args.files:
        with open(path) as fh:
            data = json.load(fh)
        n = 0
        for rec in data:
            imgs = rec.get("images") or []
            for i, p in enumerate(imgs):
                if p.startswith(args.old_prefix):
                    imgs[i] = args.new_prefix + p[len(args.old_prefix):]
                    n += 1
        if n:
            if not args.no_backup:
                shutil.copy2(path, path + ".bak")
            with open(path, "w") as fh:
                json.dump(data, fh, ensure_ascii=False)
        line = f"{path}: rewrote {n} image paths"
        if args.check:
            paths = [p for rec in data for p in (rec.get("images") or [])]
            found = sum(os.path.exists(p) for p in paths)
            line += f"  ({found}/{len(paths)} exist on disk)"
        print(line)
        total_rewritten += n
    print(f"done: {total_rewritten} paths rewritten across {len(args.files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

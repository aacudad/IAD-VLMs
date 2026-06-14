"""
Compile individual v4 trace files into grpo_train.json and sft_train.json,
matching the format of 3k_realiad_v4/ but with generated reasoning traces.

Output: 3k_realiad_v4_compiled/grpo_train.json and 3k_realiad_v4_compiled/sft_train.json
"""

import json
import os
from pathlib import Path

BASE = Path(__file__).parent
TRACE_DIR = BASE / "data" / "output" / "reasoning_traces_realiad_v4"
SOURCE_DIR = BASE / "3k_realiad_v4"
OUTPUT_DIR = BASE / "3k_realiad_v4_compiled"
IMAGES_ROOT = BASE / "data" / "Real-IAD" / "images"

_GEN_MARKER = "reasoning_traces_gen/"


def _linux_to_local(linux_path: str) -> Path:
    idx = linux_path.find(_GEN_MARKER)
    if idx == -1:
        raise ValueError(f"Cannot resolve: {linux_path}")
    rel = linux_path[idx + len(_GEN_MARKER):].replace("/", os.sep)
    return BASE / rel


def _derive_image_id(img_path: Path) -> str:
    rel = img_path.relative_to(IMAGES_ROOT)
    return "_".join(list(rel.parts[:-1]) + [rel.stem])


def load_traces(subdir: str) -> dict:
    trace_path = TRACE_DIR / subdir
    traces = {}
    for f in trace_path.glob("trace_*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            traces[data["image_id"]] = data["reasoning"]
    return traces


def compile_grpo():
    with open(SOURCE_DIR / "grpo_train.json", "r", encoding="utf-8") as f:
        original = json.load(f)

    traces = load_traces("grpo")
    print(f"GRPO: {len(traces)} traces, {len(original)} original entries")

    output = []
    missing = 0
    for entry in original:
        iid = entry["image_id"]
        if iid not in traces:
            missing += 1
            continue
        output.append({
            "image_id": iid,
            "image_path": entry["image_path"],
            "product": entry["product"],
            "is_anomaly": entry["is_anomaly"],
            "question": entry["question"],
            "answer": traces[iid],
            "gt_label": entry["gt_label"],
        })

    print(f"  Matched: {len(output)}, Missing: {missing}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "grpo_train.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  -> {OUTPUT_DIR / 'grpo_train.json'}")


def compile_sft():
    with open(SOURCE_DIR / "sft_train.json", "r", encoding="utf-8") as f:
        original = json.load(f)

    traces = load_traces("sft")
    print(f"SFT:  {len(traces)} traces, {len(original)} original entries")

    output = []
    missing = 0
    for entry in original:
        img_linux = entry["images"][0]
        try:
            img_local = _linux_to_local(img_linux)
            image_id = _derive_image_id(img_local)
        except (ValueError, Exception):
            missing += 1
            continue

        if image_id not in traces:
            missing += 1
            continue

        messages = []
        for msg in entry["messages"]:
            if msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": traces[image_id]})
            else:
                messages.append(msg)

        output.append({
            "messages": messages,
            "images": entry["images"],
        })

    print(f"  Matched: {len(output)}, Missing: {missing}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "sft_train.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  -> {OUTPUT_DIR / 'sft_train.json'}")


if __name__ == "__main__":
    compile_grpo()
    print()
    compile_sft()
    print("\nDone!")

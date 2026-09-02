"""
Full-scale Real-IAD reasoning trace generation v4.

Changes vs. v3:
  - Uses inspector_prompt_test_v2.txt (XML trace format: <think>/<location>/<type>/<answer>)
  - Word count: 120-200 words in <think> block
  - Reads from 3k_realiad_v4/ (C1-only, all 30 products, stratified)
  - Output dirs:
      data/output/reasoning_traces_realiad_v4/grpo/
      data/output/reasoning_traces_realiad_v4/sft/

Usage:
  python generate_realiad_traces_v4.py --shard_id 0 --total_shards 4
  python generate_realiad_traces_v4.py --debug
"""
import json
import logging
import os
import re
import sys
import time
import uuid
import io
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig, ThinkingConfig, ThinkingLevel
import config_mmad as config
import utils_realiad as utils

load_dotenv()

# ---------------------------------------------------------------------------
# Vertex AI credentials — service account key
# ---------------------------------------------------------------------------
# Both come from the environment, see .env.example. Nothing secret lives in this file.
_KEY_FILE = Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                 or Path(__file__).parent / "vertex-service-account.json")
if _KEY_FILE.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_KEY_FILE)
else:
    raise FileNotFoundError(f"Service account key not found: {_KEY_FILE}")

VERTEX_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
if not VERTEX_PROJECT:
    raise SystemExit("Set GOOGLE_CLOUD_PROJECT (and GOOGLE_APPLICATION_CREDENTIALS); see .env.example")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REALIAD_ROOT  = config.DATA_DIR / "Real-IAD"
IMAGES_ROOT   = REALIAD_ROOT / "images"
BASE_OUT      = config.DATA_DIR / "output" / "reasoning_traces_realiad_v4"
GRPO_DIR      = BASE_OUT / "grpo"
SFT_DIR       = BASE_OUT / "sft"

GRPO_DIR.mkdir(parents=True, exist_ok=True)
SFT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = config.BATCH_SIZE  # 10
GEMINI_MODEL = "google/gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(shard_id=None):
    suffix = str(shard_id) if shard_id is not None else "main"
    log_file = config.LOGS_DIR / f"realiad_v4_{suffix}.log"
    fmt = (
        f"%(asctime)s [Shard {shard_id}] %(levelname)s %(message)s"
        if shard_id is not None
        else "%(asctime)s %(levelname)s %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ReasoningTrace(BaseModel):
    image_id: str
    reasoning: str


class BatchReasoningTraces(BaseModel):
    traces: List[ReasoningTrace]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_system_prompt() -> str:
    prompt_file = config.PROMPTS_DIR / "inspector_prompt_test_v2.txt"
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt not found: {prompt_file}")
        sys.exit(1)


_GEN_MARKER = "reasoning_traces_gen/"

def _linux_to_local(linux_path: str) -> Path:
    """Convert a Linux server path to the local Windows equivalent."""
    idx = linux_path.find(_GEN_MARKER)
    if idx == -1:
        raise ValueError(f"Cannot resolve server path: {linux_path}")
    rel = linux_path[idx + len(_GEN_MARKER):].replace("/", os.sep)
    return Path(__file__).parent / rel


def get_normal_reference_image(category: str, camera_hint: Optional[int] = None) -> Optional[Path]:
    """Return a random OK image from the same category, preferring the same camera angle."""
    ok_root = IMAGES_ROOT / category / "OK"
    if not ok_root.exists():
        return None
    all_ok = list(ok_root.rglob("*.jpg"))
    if not all_ok:
        return None
    if camera_hint is not None:
        cam_tag = f"_C{camera_hint}_"
        matching = [p for p in all_ok if cam_tag in p.name]
        if matching:
            return random.choice(matching)
    return random.choice(all_ok)


def _derive_image_id(img_path: Path) -> str:
    """Derive a stable image_id from a path relative to IMAGES_ROOT."""
    try:
        rel = img_path.relative_to(IMAGES_ROOT)
        return "_".join(list(rel.parts[:-1]) + [rel.stem])
    except ValueError:
        return img_path.stem


def load_grpo_questions():
    """Return (ng_questions, ok_questions) from 3k_realiad_v4/grpo_train.json."""
    grpo_file = Path(__file__).parent / "3k_realiad_v4" / "grpo_train.json"
    with open(grpo_file, encoding="utf-8") as f:
        entries = json.load(f)

    ng, ok = [], []
    skipped = 0
    for entry in entries:
        try:
            img_path = _linux_to_local(entry["image_path"])
        except ValueError:
            skipped += 1
            continue
        if not img_path.exists():
            skipped += 1
            continue

        cam_match = re.search(r"_C(\d+)_", img_path.name)
        base = {
            "image_id":    entry["image_id"],
            "image_path":  img_path,
            "category":    entry["product"],
            "camera_hint": int(cam_match.group(1)) if cam_match else None,
        }

        if entry["is_anomaly"]:
            mask_path = img_path.with_suffix(".png")
            if not mask_path.exists():
                skipped += 1
                continue
            try:
                defect_type = img_path.relative_to(IMAGES_ROOT).parts[2]
            except (ValueError, IndexError):
                defect_type = "Unknown"
            ng.append({**base, "mask_path": mask_path, "defect_type": defect_type})
        else:
            ok.append({**base, "mask_path": None, "defect_type": "None"})

    logger.info(f"grpo: {len(ng)} NG + {len(ok)} OK  (skipped {skipped})")
    return ng, ok


def load_sft_questions():
    """Return (ng_questions, ok_questions) from 3k_realiad_v4/sft_train.json."""
    sft_file = Path(__file__).parent / "3k_realiad_v4" / "sft_train.json"
    with open(sft_file, encoding="utf-8") as f:
        entries = json.load(f)

    ng, ok = [], []
    skipped = 0
    for entry in entries:
        linux_path = entry["images"][0]
        try:
            img_path = _linux_to_local(linux_path)
        except ValueError:
            skipped += 1
            continue
        if not img_path.exists():
            skipped += 1
            continue

        image_id = _derive_image_id(img_path)
        cam_match = re.search(r"_C(\d+)_", img_path.name)
        try:
            category = img_path.relative_to(IMAGES_ROOT).parts[0]
        except (ValueError, IndexError):
            category = "unknown"

        base = {
            "image_id":    image_id,
            "image_path":  img_path,
            "category":    category,
            "camera_hint": int(cam_match.group(1)) if cam_match else None,
        }

        if "/NG/" in linux_path:
            mask_path = img_path.with_suffix(".png")
            if not mask_path.exists():
                ok.append({**base, "mask_path": None, "defect_type": "None"})
                continue
            try:
                defect_type = img_path.relative_to(IMAGES_ROOT).parts[2]
            except (ValueError, IndexError):
                defect_type = "Unknown"
            ng.append({**base, "mask_path": mask_path, "defect_type": defect_type})
        else:
            ok.append({**base, "mask_path": None, "defect_type": "None"})

    logger.info(f"sft:  {len(ng)} NG + {len(ok)} OK  (skipped {skipped})")
    return ng, ok


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def open_rgb(path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def save_trace(image_id: str, reasoning: str, output_dir: Path):
    out = output_dir / f"trace_{image_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"image_id": image_id, "reasoning": reasoning}, f, indent=2)


def parse_json_response(text: str) -> Dict:
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return json.loads(m.group(1) if m else text)


def normalise(d) -> Dict:
    if isinstance(d, list):
        d = {"traces": d}
    if "traces" not in d and "image_id" in d:
        d = {"traces": [d]}
    return d


# ---------------------------------------------------------------------------
# Prompt builders (v4 — XML trace format, 120-200 words)
# ---------------------------------------------------------------------------
def format_anomaly_prompt(batch: List[Dict]) -> str:
    parts = [
        f"You will receive {len(batch)} samples to inspect.",
        "For EACH sample, you are provided with THREE images in the order they appear.",
        "",
        "Inspect each sample visually and generate a complete inspection record "
        "following the specified format.",
        "",
        "IMPORTANT: Each <think> block must contain 120-200 words of detailed "
        "visual reasoning. Do not abbreviate.",
        "",
        "CRITICAL: Your reasoning must read as a single-image visual inspection. "
        "Do not mention, imply, or reference the existence of multiple images, "
        "reference images, overlays, highlighted regions, or comparisons with "
        "other views. Write as though you are inspecting one product image only.",
        "",
    ]
    for idx, item in enumerate(batch, 1):
        parts.append(f"--- SAMPLE {idx} ---")
        parts.append(f"Image ID: {item['image_id']}")
        parts.append(f"Product Type: {item['category']}")
        parts.append(f"[INTERNAL — do not reference in reasoning] Ground Truth: ANOMALOUS")
        if item.get('defect_type'):
            parts.append(f"[INTERNAL] Defect Type: {item['defect_type']}")
        parts.append(
            "[INTERNAL] The red overlay (second image) highlights the region of interest. "
            "For the <location> tag, use a 3x3 grid: top-left, top-center, top-right, "
            "middle-left, center, middle-right, bottom-left, bottom-center, bottom-right. "
            "If the defect spans multiple regions, separate them with commas "
            "(e.g. top-center, top-left)."
        )
        parts.append("")
    parts.append("Return a JSON object with a 'traces' array containing one entry per sample.")
    return "\n".join(parts)


def format_normal_prompt(batch: List[Dict]) -> str:
    parts = [
        f"You will receive {len(batch)} samples to inspect.",
        "For EACH sample, you are provided with ONE image: the original product image.",
        "",
        "Inspect each sample visually and generate a complete inspection record "
        "following the specified format.",
        "",
        "IMPORTANT: Each <think> block must contain 120-200 words of detailed "
        "visual reasoning. Do not abbreviate.",
        "",
    ]
    for idx, item in enumerate(batch, 1):
        parts.append(f"--- SAMPLE {idx} ---")
        parts.append(f"Image ID: {item['image_id']}")
        parts.append(f"Product Type: {item['category']}")
        parts.append(f"[INTERNAL — do not reference in reasoning] Ground Truth: NORMAL")
        parts.append("")
    parts.append("Return a JSON object with a 'traces' array containing one entry per sample.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Gemini batch call
# ---------------------------------------------------------------------------
def process_batch(
    batch: List[Dict],
    client,
    model: str,
    system_prompt: str,
    normal: bool = False,
    debug: bool = False,
    shard_id: int = 0,
) -> Dict[str, str]:
    batch_id = str(uuid.uuid4())[:8]
    text = format_normal_prompt(batch) if normal else format_anomaly_prompt(batch)
    parts = [Part.from_text(text=text)]

    debug_dir = Path("debug_output")
    prefix = f"shard{shard_id}_{'normal' if normal else 'anomaly'}"

    if debug:
        debug_dir.mkdir(exist_ok=True)
        with open(debug_dir / f"{prefix}_batch_{batch_id}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{text}")

    for i, item in enumerate(batch):
        img = open_rgb(item["image_path"])
        parts.append(Part.from_bytes(data=image_to_jpeg_bytes(img), mime_type="image/jpeg"))

        if not normal and item.get("mask_path"):
            overlay = utils.create_overlay(item["image_path"], item["mask_path"]) or img
            parts.append(Part.from_bytes(data=image_to_jpeg_bytes(overlay), mime_type="image/jpeg"))

            ref_path = get_normal_reference_image(item["category"], item.get("camera_hint"))
            if ref_path:
                ref = open_rgb(ref_path)
                parts.append(Part.from_bytes(data=image_to_jpeg_bytes(ref), mime_type="image/jpeg"))
                if debug:
                    ref.save(debug_dir / f"{prefix}_batch_{batch_id}_s{i}_ref.jpg")
            else:
                logger.warning(f"No normal reference found for category '{item['category']}'")

            if debug:
                debug_dir.mkdir(exist_ok=True)
                overlay.save(debug_dir / f"{prefix}_batch_{batch_id}_s{i}_overlay.jpg")

        if debug:
            debug_dir.mkdir(exist_ok=True)
            img.save(debug_dir / f"{prefix}_batch_{batch_id}_s{i}_orig.jpg")

    if debug:
        logger.info(f"[DEBUG] Batch {batch_id} saved (no API call).")
        return {item["image_id"]: "DEBUG_TRACE" for item in batch}

    cfg = GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[Part.from_text(text=system_prompt)],
        thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL),
    )
    full = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=[Content(role="user", parts=parts)],
        config=cfg,
    ):
        if chunk.text:
            full += chunk.text

    validated = BatchReasoningTraces(**normalise(parse_json_response(full)))
    return {t.image_id: t.reasoning for t in validated.traces}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def run(
    shard_id: int = 0,
    total_shards: int = 1,
    normal_only: bool = False,
    anomaly_only: bool = False,
    debug: bool = False,
):
    global logger
    logger = setup_logging(shard_id)
    logger.info(f"=== Real-IAD v4 Generation | Shard {shard_id+1}/{total_shards} ===")
    if debug:
        logger.warning("DEBUG mode — no API calls.")

    system_prompt = load_system_prompt()
    logger.info(f"Vertex AI project={VERTEX_PROJECT}, location={VERTEX_LOCATION}, model={GEMINI_MODEL}")

    def _run_type(questions: List[Dict], output_dir: Path, normal: bool):
        label = "normal" if normal else "anomaly"

        # Sharding: this shard processes every Nth question
        shard_questions = [q for i, q in enumerate(questions) if i % total_shards == shard_id]
        logger.info(f"[{label}] Shard assigned {len(shard_questions)} / {len(questions)} total")

        # Skip already done
        processed = set()
        for f in output_dir.glob("trace_*.json"):
            if f.stem.startswith("trace_"):
                processed.add(f.stem[6:])
        logger.info(f"[{label}] Already processed: {len(processed)}")

        remaining = [q for q in shard_questions if q["image_id"] not in processed]
        logger.info(f"[{label}] To process: {len(remaining)}")
        if not remaining:
            logger.info(f"[{label}] Nothing to do.")
            return

        client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        max_retries = config.MAX_RETRIES * 3

        attempt_counts: Dict[str, int] = {}
        permanently_skipped: set = set()

        with tqdm(total=len(remaining), desc=f"Shard{shard_id} {label}") as pbar:
            while True:
                todo = [
                    q for q in remaining
                    if q["image_id"] not in processed
                    and q["image_id"] not in permanently_skipped
                ]
                if not todo:
                    break

                for q in todo:
                    if attempt_counts.get(q["image_id"], 0) >= config.MAX_RETRIES:
                        if q["image_id"] not in permanently_skipped:
                            logger.warning(f"Skipping {q['image_id']} (max retries)")
                            permanently_skipped.add(q["image_id"])

                batch = []
                for q in todo:
                    if len(batch) >= BATCH_SIZE:
                        break
                    if q["image_id"] not in permanently_skipped:
                        batch.append(q)
                        attempt_counts[q["image_id"]] = attempt_counts.get(q["image_id"], 0) + 1
                if not batch:
                    break

                success = False
                retry_count = 0
                _max = 1 if debug else max_retries

                while not success and retry_count < _max:
                    try:
                        traces = process_batch(
                            batch, client, GEMINI_MODEL, system_prompt,
                            normal=normal, debug=debug, shard_id=shard_id,
                        )
                        if not debug:
                            saved = 0
                            for item in batch:
                                if item["image_id"] in traces:
                                    save_trace(item["image_id"], traces[item["image_id"]], output_dir)
                                    processed.add(item["image_id"])
                                    saved += 1
                            pbar.update(saved)
                        else:
                            for item in batch:
                                processed.add(item["image_id"])
                            pbar.update(len(batch))
                        success = True

                    except Exception as e:
                        if debug:
                            logger.error(f"[DEBUG] Error: {e}")
                            break

                        err = str(e)
                        if "429" in err or "RESOURCE_EXHAUSTED" in err:
                            logger.warning(f"Rate limit. Waiting 10s...")
                            time.sleep(10)
                        else:
                            logger.error(f"Batch error: {e}")
                            time.sleep(5)

                        retry_count += 1

        logger.info(f"[{label}] Done. Output: {output_dir}")

    grpo_ng, grpo_ok = load_grpo_questions()
    sft_ng,  sft_ok  = load_sft_questions()

    # 4 passes: grpo-NG, grpo-OK, sft-NG, sft-OK
    if not normal_only:
        _run_type(grpo_ng, GRPO_DIR, normal=False)
    if not anomaly_only:
        _run_type(grpo_ok, GRPO_DIR, normal=True)
    if not normal_only:
        _run_type(sft_ng,  SFT_DIR,  normal=False)
    if not anomaly_only:
        _run_type(sft_ok,  SFT_DIR,  normal=True)

    logger.info("=== Shard complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Real-IAD v4 reasoning traces (Gemini only, inspector_prompt_test_v2, C1 images, 120-200 words)."
    )
    parser.add_argument("--shard_id",     type=int, default=0,  help="0-based shard index")
    parser.add_argument("--total_shards", type=int, default=1,  help="Total number of shards")
    parser.add_argument("--anomaly_only", action="store_true",  help="Only process anomalous images")
    parser.add_argument("--normal_only",  action="store_true",  help="Only process normal images")
    parser.add_argument("--debug",        action="store_true",  help="Debug: save prompts/images, skip API calls")
    args = parser.parse_args()

    run(
        shard_id=args.shard_id,
        total_shards=args.total_shards,
        normal_only=args.normal_only,
        anomaly_only=args.anomaly_only,
        debug=args.debug,
    )

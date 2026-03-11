"""
Bio-VQA pipeline
Extracts figures from PMC .tar.gz archives and runs VQA models on them.
"""

import json
import logging
import os
import re
import shutil
import statistics
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from bio_vqa.models import Figure, VQAResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = {
    # Free Colab T4 (15 GB VRAM)
    "qwen2.5-vl-3b":   {"hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",       "type": "qwen2vl",   "vram": 8,  "t4": True},
    "paligemma-3b":    {"hf_id": "google/paligemma-3b-mix-448",          "type": "paligemma", "vram": 7,  "t4": True},
    "internvl2-2b":    {"hf_id": "OpenGVLab/InternVL2-2B",              "type": "internvl",  "vram": 5,  "t4": True},
    "llava-1.5-7b":    {"hf_id": "llava-hf/llava-1.5-7b-hf",            "type": "llava",     "vram": 14, "t4": True},
    # Needs A100
    "qwen2.5-vl-7b":   {"hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",        "type": "qwen2vl",   "vram": 16, "t4": False},
    "internvl2-8b":    {"hf_id": "OpenGVLab/InternVL2-8B",              "type": "internvl",  "vram": 18, "t4": False},
    # Captioning only (not real VQA)
    "florence-2-large":{"hf_id": "microsoft/Florence-2-large",           "type": "florence2", "vram": 4,  "t4": True},
    # Legacy baselines
    "blip2-opt-2.7b":  {"hf_id": "Salesforce/blip2-opt-2.7b",           "type": "blip2",     "vram": 6,  "t4": True},
    "blip-vqa-base":   {"hf_id": "Salesforce/blip-vqa-base",             "type": "blip",      "vram": 1,  "t4": True},
    # API
}

DEFAULT_MODEL = "blip-vqa-base"

QUESTIONS = [
    {
        "type": "figure_understanding",
        "question": "How many panels are in this figure? Label each panel (A, B, C...) and state what type of plot each shows (e.g. chromatogram, mass spectrum, bar chart, scatter plot).",
        "use_context": False,
    },
    {
        "type": "text_qa",
        "question": "Read the axis labels from this figure. What quantity is on the x-axis and what quantity is on the y-axis? Include units if shown.",
        "use_context": False,
    },
    {
        "type": "figure_understanding",
        "question": "Identify the most prominent peaks or features in this figure. Report their x-axis positions and any numeric labels shown on or near the peaks.",
        "use_context": False,
    },
    {
        "type": "compound_id",
        "question": "Are any compound names, compound numbers, or chemical identifiers written on this figure? List all visible labels.",
        "use_context": False,
    },
    {
        "type": "table_reasoning",
        "question": "What are the minimum and maximum values on each axis? If a table is present, summarise the numerical ranges in each column.",
        "use_context": False,
    },
    {
        "type": "cross_modal",
        "question": "Given the article context, what compounds or compound classes are being detected in this figure? What analytical technique is used?",
        "use_context": True,
    },
    {
        "type": "cross_modal",
        "question": "Based on the article context and this figure, what is the main scientific conclusion? How does it support the paper's findings?",
        "use_context": True,
    },
]


def list_models():
    """Print all models."""
    print(f"\n{'Key':<22} {'VRAM':>5}  {'T4?':>4}  HuggingFace ID")
    print("-" * 72)
    for key, cfg in MODELS.items():
        t4 = "yes" if cfg["t4"] else "no"
        hf = cfg["hf_id"] or "API"
        tag = "  <- default" if key == DEFAULT_MODEL else ""
        print(f"{key:<22} {cfg['vram']:>4}GB  {t4:>4}  {hf}{tag}")


# ---------------------------------------------------------------------------
# Step 1: Parse PMC archive
# ---------------------------------------------------------------------------

def _get_text(el) -> str:
    """Recursively get plain text from an XML element."""
    parts = []
    if el.text:
        parts.append(el.text.strip())
    for child in el:
        parts.append(_get_text(child))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


def parse_archive(archive_path: str, work_dir: str) -> list[Figure]:
    """Extract a PMC .tar.gz and return a list of Figure objects."""
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(work_dir)

    xml_files = list(Path(work_dir).rglob("*.nxml"))
    if not xml_files:
        raise FileNotFoundError(f"No .nxml file found in {archive_path}")

    xml_path = xml_files[0]
    article_dir = xml_path.parent
    root = ET.parse(xml_path).getroot()

    title_el = root.find(".//{*}article-title")
    article_title = _get_text(title_el) if title_el is not None else ""

    abstract_el = root.find(".//{*}abstract")
    article_abstract = _get_text(abstract_el) if abstract_el is not None else ""

    figures = []
    for el in root.iter():
        if not (el.tag.endswith("}fig") or el.tag == "fig"):
            continue

        fig_id = el.get("id", "unknown")

        label_el = el.find(".//{*}label") or el.find(".//label")
        label = label_el.text.strip() if (label_el is not None and label_el.text) else fig_id

        caption_el = el.find(".//{*}caption") or el.find(".//caption")
        caption = _get_text(caption_el) if caption_el is not None else ""

        graphic_el = el.find(".//{*}graphic") or el.find(".//graphic")
        if graphic_el is None:
            continue

        href = graphic_el.get("{http://www.w3.org/1999/xlink}href", "")
        if not href:
            continue

        # PMC archives use bare stems as hrefs — try a few extensions
        base = Path(href).stem
        jpg = None
        for cand in [article_dir / f"{base}.jpg", article_dir / f"{href}.jpg", article_dir / f"{base}.jpeg"]:
            if cand.exists():
                jpg = str(cand)
                break

        if jpg is None:
            logger.warning("No image found for %s (href=%s)", fig_id, href)
            continue

        figures.append(Figure(
            figure_id=fig_id, label=label, caption=caption,
            image_path=jpg, filename=Path(jpg).name,
            article_title=article_title, article_abstract=article_abstract,
        ))

    logger.info("Found %d figure(s) in %s", len(figures), xml_path.name)
    return figures


# ---------------------------------------------------------------------------
# Step 2: Load model
# ---------------------------------------------------------------------------

def load_model(model_key: str, hf_token: Optional[str] = None):
    """Load a VQA model. Returns (processor, model, model_type, device)."""
    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration, LlavaForConditionalGeneration
    from huggingface_hub import snapshot_download

    if model_key not in MODELS:
        raise KeyError(f"Unknown model '{model_key}'. Run list_models() to see options.")

    cfg = MODELS[model_key]
    hf_id, mtype = cfg["hf_id"], cfg["type"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if mtype == "claude_api":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY environment variable to use Claude.")
        return api_key, None, mtype, "api"

    # Download with retry — HF CDN returns 429s under load
    for attempt in range(5):
        try:
            snapshot_download(hf_id, token=token, ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"])
            break
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = 30 * (2 ** attempt)
                logger.warning("HF rate limit, waiting %ds (attempt %d/5)...", wait, attempt + 1)
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError(f"Could not download {hf_id} after 5 attempts.")

    logger.info("Loading %s on %s...", model_key, device)

    if mtype == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, token=token)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
            trust_remote_code=True, token=token)

    elif mtype == "internvl":
        from transformers import AutoModel, AutoTokenizer
        processor = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=token)
        model = AutoModel.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True, trust_remote_code=True, token=token)
        model.eval()

    elif mtype == "paligemma":
        from transformers import PaliGemmaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(hf_id, token=token)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None, token=token)

    elif mtype == "llava":
        processor = AutoProcessor.from_pretrained(hf_id, token=token)
        model = LlavaForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None, token=token)

    elif mtype == "florence2":
        from transformers import AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
            trust_remote_code=True, token=token)

    elif mtype == "blip2":
        processor = AutoProcessor.from_pretrained(hf_id, token=token)
        model = Blip2ForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None, token=token)

    elif mtype == "blip":
        from transformers import BlipForQuestionAnswering, BlipProcessor
        processor = BlipProcessor.from_pretrained(hf_id)
        model = BlipForQuestionAnswering.from_pretrained(hf_id, torch_dtype=dtype).to(device)

    else:
        raise ValueError(f"Unknown model type: {mtype}")

    logger.info("Loaded %s", model_key)
    return processor, model, mtype, device


# ---------------------------------------------------------------------------
# Step 3: Run VQA inference
# ---------------------------------------------------------------------------

def run_vqa(processor, model, model_type: str, device: str, image, question: str, context: Optional[str] = None) -> str:
    """Run a single VQA forward pass and return the answer string."""
    import torch

    prompt = f"Context: {context[:500]}\n\nQuestion: {question}" if context else question

    if model_type == "claude_api":
        import base64, io, json, urllib.request
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt}
            ]}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type": "application/json", "x-api-key": processor,
                     "anthropic-version": "2023-06-01"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["content"][0]["text"].strip()

    if model_type == "qwen2vl":
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    if model_type == "internvl":
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        pixel_values = transform(image).unsqueeze(0).to(device, dtype=torch.float16)
        response = model.chat(processor, pixel_values, f"<image>\n{prompt}", dict(max_new_tokens=512, do_sample=False))
        return response.strip()

    if model_type == "florence2":
        # Florence-2 can't answer open questions — run fixed task tokens and combine
        parts = []
        for task, label in [("<MORE_DETAILED_CAPTION>", "Description"), ("<OCR>", "OCR text")]:
            inp = processor(text=task, images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                     max_new_tokens=512, num_beams=3, early_stopping=False)
            raw = processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(raw, task=task, image_size=(image.width, image.height))
            text = str(parsed.get(task, "")).strip()
            if text:
                parts.append(f"{label}: {text}")
        try:
            inp2 = processor(text="<DENSE_REGION_CAPTION>", images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                out2 = model.generate(input_ids=inp2["input_ids"], pixel_values=inp2["pixel_values"],
                                      max_new_tokens=512, num_beams=3, early_stopping=False)
            raw2 = processor.batch_decode(out2, skip_special_tokens=False)[0]
            parsed2 = processor.post_process_generation(raw2, task="<DENSE_REGION_CAPTION>",
                                                        image_size=(image.width, image.height))
            labels = parsed2.get("<DENSE_REGION_CAPTION>", {})
            if isinstance(labels, dict) and labels.get("labels"):
                parts.append(f"Region labels: {', '.join(labels['labels'][:20])}")
        except Exception:
            pass
        if not parts:
            return "[Florence-2: no output]"
        combined = "\n".join(parts)
        q = question.lower()
        if any(w in q for w in ["axis", "label", "unit", "x-axis", "y-axis"]):
            for p in parts:
                if p.startswith("OCR"):
                    return p
        if any(w in q for w in ["compound", "name", "number", "labeled"]):
            for p in parts:
                if p.startswith("Region"):
                    return p
        return combined

    if model_type == "paligemma":
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)
        full = processor.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):].strip() if full.startswith(prompt) else full.strip()

    if model_type == "blip":
        # BLIP was designed for short VQA questions (max ~512 tokens total).
        inputs = processor(image, question, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    if model_type == "blip2":
        vqa_prompt = f"Context: {context[:300]} Question: {question} Answer:" if context else f"Question: {question} Answer:"
        inputs = processor(image, vqa_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False, num_beams=5)
        return processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    if model_type == "llava":
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)
        full = processor.decode(out[0], skip_special_tokens=True)
        return full.split("[/INST]")[-1].strip() if "[/INST]" in full else full.strip()

    raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Step 4: Evaluate
# ---------------------------------------------------------------------------

_STOP = {"the","a","an","is","are","was","were","of","in","to","and","or",
         "that","this","it","for","with","as","at","from","by","on","be",
         "has","have","not","no","i","we"}

def _tok(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP

def caption_overlap(answer: str, caption: str) -> float:
    """Token F1 between answer and caption — proxy for answer quality."""
    a, c = _tok(answer), _tok(caption)
    if not a or not c:
        return 0.0
    p = len(a & c) / len(a)
    r = len(a & c) / len(c)
    return round(2 * p * r / (p + r), 4) if p + r else 0.0

def completeness(answer: str) -> float:
    """Rough completeness score based on length and punctuation."""
    w = len(answer.split())
    return round(min(w / 30, 1.0) + 0.05 * int(any(c in answer for c in ".,:;")), 4)

def cross_model_agreement(answers: list[str]) -> float:
    """Mean pairwise Jaccard similarity across model answers for the same question."""
    if len(answers) < 2:
        return 1.0
    sets = [_tok(a) for a in answers]
    scores = [len(sets[i] & sets[j]) / len(sets[i] | sets[j])
              for i in range(len(sets)) for j in range(i + 1, len(sets))
              if sets[i] | sets[j]]
    return round(statistics.mean(scores), 4) if scores else 0.0

def evaluate(results: list[VQAResult], figures: list[Figure]) -> list[VQAResult]:
    """Fill in evaluation scores on each result."""
    fig_map = {f.figure_id: f for f in figures}

    groups = defaultdict(list)
    for r in results:
        groups[(r.figure_id, r.question)].append(r.answer)
    agreement_map = {k: cross_model_agreement(v) for k, v in groups.items()}

    for r in results:
        cap = fig_map.get(r.figure_id, Figure("","","","","")).caption
        r.caption_overlap = caption_overlap(r.answer, cap)
        r.completeness = completeness(r.answer)
        r.cross_model_agreement = agreement_map[(r.figure_id, r.question)]
        r.composite_score = round((r.caption_overlap + r.completeness) / 2, 4)

    return results


# ---------------------------------------------------------------------------
# Step 5: Run everything
# ---------------------------------------------------------------------------

def run_pipeline(
    archive_path: str,
    model_keys: Optional[list[str]] = None,
    questions: Optional[list[dict]] = None,
    output_dir: str = "vqa_output",
    max_figures: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> dict:
    """
    Full pipeline: parse archive -> run VQA -> evaluate -> save results.
    Returns the results dict (also saved to output_dir/results.json).
    """
    from PIL import Image as PILImage

    model_keys = model_keys or [DEFAULT_MODEL]
    questions = questions or QUESTIONS
    os.makedirs(output_dir, exist_ok=True)

    # 1. Parse
    with tempfile.TemporaryDirectory() as tmp:
        figures = parse_archive(archive_path, tmp)
        if max_figures:
            figures = figures[:max_figures]
        for fig in figures:
            dest = os.path.join(output_dir, fig.filename)
            shutil.copy2(fig.image_path, dest)
            fig.image_path = dest

    if not figures:
        logger.error("No figures found.")
        return {}

    # 2. Inference
    all_results = []
    for mk in model_keys:
        if mk not in MODELS:
            logger.warning("Unknown model '%s', skipping.", mk)
            continue
        try:
            processor, model, mtype, device = load_model(mk, hf_token=hf_token)
        except Exception as e:
            logger.error("Could not load %s: %s", mk, e)
            continue

        for fig in figures:
            try:
                img = PILImage.open(fig.image_path).convert("RGB")
            except Exception as e:
                logger.warning("Cannot open %s: %s", fig.image_path, e)
                continue

            for q in questions:
                context = f"{fig.article_title}. {fig.article_abstract}" if q.get("use_context") else None
                t0 = time.perf_counter()
                try:
                    answer = run_vqa(processor, model, mtype, device, img, q["question"], context)
                except Exception as e:
                    answer = f"[ERROR: {e}]"
                all_results.append(VQAResult(
                    figure_id=fig.figure_id, question=q["question"],
                    question_type=q.get("type", "general"), model_name=mk,
                    answer=answer, latency_s=round(time.perf_counter() - t0, 3),
                    context_used=bool(q.get("use_context")),
                ))
                logger.info("[%s] %s | %s: %s", mk, fig.figure_id, q.get("type"), answer[:80])

        # Free GPU memory before loading next model
        try:
            import torch; del model; torch.cuda.empty_cache()
        except Exception:
            pass

    # 3. Evaluate
    all_results = evaluate(all_results, figures)

    # 4. Save
    payload = {
        "archive": archive_path,
        "figures": [asdict(f) for f in figures],
        "results": [asdict(r) for r in all_results],
    }
    with open(os.path.join(output_dir, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    # 5. Print leaderboard
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r.model_name].append(r)

    rows = [(mk, statistics.mean(r.caption_overlap for r in rs),
             statistics.mean(r.completeness for r in rs),
             statistics.mean(r.latency_s for r in rs))
            for mk, rs in by_model.items()]
    rows.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("  BIO-VQA RESULTS")
    print("=" * 70)
    print(f"  {'Model':<28} {'Caption F1':>10} {'Complete':>10} {'Latency':>10}")
    print("  " + "-" * 64)
    for mk, cap, comp, lat in rows:
        print(f"  {mk:<28} {cap:>10.4f} {comp:>10.4f} {lat:>9.2f}s")
    print("=" * 70)

    # 6. Write markdown report
    lines = ["# Bio-VQA Report", f"\nArchive: `{archive_path}`  ",
             f"Figures: {len(figures)} | Q&A pairs: {len(all_results)}\n", "---\n"]
    for fig in figures:
        lines.append(f"## {fig.label} — `{fig.filename}`")
        lines.append(f"\n**Caption:** {fig.caption}\n")
        fig_results = [r for r in all_results if r.figure_id == fig.figure_id]
        for mk in dict.fromkeys(r.model_name for r in fig_results):
            lines.append(f"### `{mk}`\n")
            lines.append("| Type | Question | Answer | Score | Latency |")
            lines.append("|---|---|---|---|---|")
            for r in fig_results:
                if r.model_name != mk:
                    continue
                q = r.question[:55] + "…" if len(r.question) > 55 else r.question
                a = r.answer[:100].replace("\n", " ") + ("…" if len(r.answer) > 100 else "")
                lines.append(f"| {r.question_type} | {q} | {a} | {r.composite_score} | {r.latency_s}s |")
            lines.append("")

    with open(os.path.join(output_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines))

    print(f"\nSaved to {output_dir}/  (results.json, report.md)")
    return payload

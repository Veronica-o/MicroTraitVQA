"""
run.py — command-line entry point for Bio-VQA

Examples:
    python run.py --archive PMC11047695_tar.gz
    python run.py --archive PMC11047695_tar.gz --models qwen2.5-vl-3b paligemma-3b
    python run.py --archive PMC11047695_tar.gz --max-figures 2
    python run.py --list-models
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Strip Jupyter kernel args so this script also works when imported in a notebook
argv = [a for a in sys.argv[1:] if not a.startswith("-f")]

from bio_vqa import list_models, run_pipeline, MODELS, DEFAULT_MODEL

parser = argparse.ArgumentParser(description="Bio-VQA: VQA on PMC figures")
parser.add_argument("--archive", help="Path to a PMC .tar.gz archive")
parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL], help="Model key(s) to use")
parser.add_argument("--questions", help="Path to a custom questions JSON file")
parser.add_argument("--output-dir", default=None,
                    help="Output directory (default: auto-generated from model/archive/time)")
parser.add_argument("--max-figures", type=int, default=None)
parser.add_argument("--list-models", action="store_true")
args = parser.parse_args(argv)

if args.list_models:
    list_models()
    sys.exit(0)

if not args.archive:
    parser.error("--archive is required (or use --list-models)")

unknown = [k for k in args.models if k not in MODELS]
if unknown:
    parser.error(f"Unknown model(s): {unknown}. Run --list-models for options.")
# Auto-generate output dir: results_<models>_<archive>_<timestamp>
if args.output_dir is None:
    models_tag = "+".join(m.replace("/", "-") for m in args.models)
    archive_tag = Path(args.archive).stem.replace(".tar", "")
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"vqa_output/results_{models_tag}_{archive_tag}_{time_tag}"

print(f"Output dir: {args.output_dir}")

custom_questions = None
if args.questions:
    with open(args.questions) as fh:
        custom_questions = json.load(fh)

run_pipeline(
    archive_path=args.archive,
    model_keys=args.models,
    questions=custom_questions,
    output_dir=args.output_dir,
    max_figures=args.max_figures,
)

# Bio-VQA

Runs Visual Question Answering models on figures extracted from PubMed Central Open Access archives.
Given a PMC `.tar.gz` it extracts the figures, runs one or more VQA models on each,
scores the answers, and saves `results.json` + `report.md`.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/bio-vqa.git
cd bio-vqa
pip install -r requirements.txt
```

## Usage

```bash
# basic run (default model: qwen2.5-vl-3b)
python run.py --archive PMC11047695_tar.gz

# run two models
python run.py --archive PMC11047695_tar.gz --models qwen2.5-vl-3b paligemma-3b

# quick test with 1 figure
python run.py --archive PMC11047695_tar.gz --max-figures 1

# see all available models
python run.py --list-models
```

Or from Python:

```python
from bio_vqa import run_pipeline

run_pipeline(
    archive_path="PMC11047695_tar.gz",
    model_keys=["qwen2.5-vl-3b"],
    output_dir="vqa_output",
)
```

## Models

### Free Colab T4 (15 GB VRAM)

| Key | VRAM | Notes |
|---|---|---|
| `qwen2.5-vl-3b` | 8 GB | **Recommended.** Best on charts and tables. |
| `paligemma-3b` | 7 GB | Good on scientific figures. [Needs HF licence.](#paligemma) |
| `internvl2-2b` | 5 GB | Smallest true VQA model. |
| `llava-1.5-7b` | 14 GB | Fits T4 (tight). |
| `florence-2-large` | 4 GB | Captioning only — not real VQA. |

### Needs A100 (40 GB)

| Key | VRAM |
|---|---|
| `qwen2.5-vl-7b` | 16 GB |
| `internvl2-8b` | 18 GB |

### Legacy / API

| Key | Notes |
|---|---|
| `blip2-opt-2.7b` | Legacy baseline |
| `blip-vqa-base` | CPU-runnable, good for testing |
| `claude-api` | Needs `ANTHROPIC_API_KEY` env var |

## Output

```
vqa_output/
  results.json   all Q&A pairs with scores
  report.md      per-figure markdown summary
  *.jpg          extracted figures
```

Scores:
- `caption_overlap` — token F1 between the answer and the figure caption (proxy for accuracy)
- `completeness` — heuristic based on answer length
- `composite_score` — mean of the above two

## Project layout

```
bio_vqa/
  models.py     Figure and VQAResult dataclasses
  vqa.py        everything else: config, parsing, inference, evaluation, pipeline
  __init__.py   public exports
run.py          CLI entry point
tests/
  test_evaluation.py  unit tests (no GPU needed)
  test_parsing.py     parsing tests (skip if no archive)
notebooks/
  colab_demo.ipynb
requirements.txt
```

## Tests

```bash
# no GPU needed
pytest tests/test_evaluation.py -v

# with a real archive in tests/data/
pytest tests/ -v
```

## Colab

Open `notebooks/colab_demo.ipynb`.
Update the `REPO` variable in the first cell to point at your fork.

---

### PaliGemma

First-time download requires accepting the licence at
[huggingface.co/google/paligemma-3b-mix-448](https://huggingface.co/google/paligemma-3b-mix-448),
then: `export HF_TOKEN=hf_your_token_here`

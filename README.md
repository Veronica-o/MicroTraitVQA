# Bio-VQA

Runs Visual Question Answering models on figures extracted from PubMed Central Open Access archives.
Given a PMC `.tar.gz` it extracts the figures, runs one or more VQA models on each,
scores the answers, and saves `results.json` + `report.md`.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/MicroTraitVQA.git
cd bio-vqa
pip install -r requirements.txt
```

## Usage

```bash
# basic run (default model: )
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




## Output

```
vqa_output/
  results.json   all Q&A pairs with scores
  report.md      per-figure markdown summary
  *.jpg          extracted figures
```

Scores:
- `Rouge Score` 
- `Bleu1 Score` 
- `Bleu4 Score` 
- `composite_score`

## Project layout

```
bio_vqa/
  models.py     Figure and VQAResult dataclasses
  vqa.py        everything else: config, parsing, inference, evaluation, pipeline
  __init__.py   public exports
run.py          CLI entry point
requirements.txt
```


```

### PaliGemma

First-time download requires accepting the licence at
[huggingface.co/google/paligemma-3b-mix-448](https://huggingface.co/google/paligemma-3b-mix-448),
then: `export HF_TOKEN=hf_your_token_here`

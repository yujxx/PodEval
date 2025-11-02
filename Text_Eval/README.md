# Text Evaluation Tools

Evaluating dialogue quality using both quantitative metrics and LLM-as-a-Judge methods.

<p align="center">
  <img align="middle" width="800" src="text-eval.png"/>
</p>

## Quick Start

### 1. Quantitative Metrics (`eval_richness_obj.py`)
Analyze text content using objective linguistic metrics:

```bash
python eval_richness_obj.py \
    --input_dir /path/to/dialogue \
    --output_dir /path/to/output
```

**Evaluation metrics**: distinct-2, information density, semantic diversity, MATTR

### 2. Subjective Evaluation (`eval_by_gpt.py`)
Compare two sets of dialogues using GPT-4:

```bash
python eval_by_gpt.py \
    --input-dir-1 /path/to/dialogue1 \
    --input-dir-2 /path/to/dialogue2 \
    --output-dir /path/to/output
```

**Evaluation metrics**: coherence, engagingness, diversity, informativeness, overall quality, speaker diversity

## Input Format

JSON files with dialogue content:

```json
{"speaker": "Speaker1", "speaking_content": "Dialogue content here"}
```

## Environment

See PodEval/README.md

- Set environment variables for GPT evaluation:
```bash
export EVAL_OPENAI_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"
```

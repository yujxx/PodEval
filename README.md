# PodEval: Comprehensive Podcast Evaluation Toolkit

A comprehensive toolkit for podcast evaluation across multiple dimensions including audio, speech, and text using both objective metrics and subjective evaluation methods.

<p align="center">
  <img align="middle" width="800" src="PodEval-icon.png"/>
</p>


## Overview

PodEval provides a complete evaluation pipeline for podcast generation systems, supporting:
- **Real-world Dataset** - Curated dataset of real podcast episodes for benchmarking
- **Text Quality Evaluation** - Both quantitative linguistic metrics and LLM-based subjective evaluation
- **Speech/Audio Assessment** - Objective speech and audio evaluation metrics and Subjective listening tests.

## Directory Structure

### 📁 [Real_Pod/](./Real_Pod/)
**Real-Pod Dataset** - A curated dataset of real-world podcast episodes serving as a reference for human-level creative quality.

- **Content**: 51 topics across 17 categories with diverse audio scenarios
- **Usage**: Download Real-Pod dataset; Process and prepare any podcast dataset for unified evaluation format.
- **Documentation**: [Real_Pod/README.md](./Real_Pod/README.md)


### 📁 [Text_Eval/](./Text_Eval/)
**Text Evaluation Tools** - Evaluate conversation scripts using quantitative metrics and LLM-as-a-Judge methods.

- **Methods**:
  - **Quantitative Metrics**: distinct-2, information density, semantic diversity, MATTR
  - **LLM-as-a-Judge**: GPT-based evaluation for dialogue, including metrics like coherence, engagingness, diversity, informativeness, overall quality, speaker diversity
- **Documentation**: [Text_Eval/README.md](./Text_Eval/README.md)


### 📁 [Speech_Audio_Objective_Evaluation/](./Speech_Audio_Obj_Eval/)
**Objective Speech/Audio Evaluation Toolkit** - Evaluate objective quality metrics of podcast audio.

- **Metrics**: DNSMOS, Loudness, WER, Speaker Similarity, Speaker Timbre Difference, Speech-to-Music Ratio, Music-Speech Harmony.
- **Documentation**: [Speech_Audio_Obj_Eval/README.md](./Speech_Audio_Obj_Eval/README.md)


### 📁 [Subjective_Listening_Tests/](./Subjective_Listening_Tests/)
**Subjective Listening Tests** - Human evaluation framework for podcast speech/audio assessment.

- **Dialogue Naturalness Evaluation**: Evaluate the naturalness and authenticity of dialogue speech in podcast.
- **Questionnaire-based MOS Test**: Comprehensive evaluation of long-form podcast content through structured questionnaires.


## Environment

```bash
conda create --name podeval python=3.10
conda activate podeval
pip install -r requirements.txt
```


## Citation

If you use PodEval in your research, please cite:

(Comming soon)

## License

See project root for license information.
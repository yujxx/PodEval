# **Real-Pod Dataset**

## **Overview**

The **Real-Pod Dataset** is designed to:
- **Provide Realistic Data**: Serve as a reference for evaluating podcast creation systems with real-world data.
- **Cover Diverse Topics**: Include a wide range of podcast categories and topics to ensure the dataset reflects the variety of content available in the podcast ecosystem.
- **Enable Comprehensive Evaluation**: Support the evaluation of speech/audio quality, dialogue naturalness, music and sound effect (MSE) harmony and so on.

<p align="center">
  <img align="middle" width="800" src="Figure_dataset.png"/>
</p>

---

## **Table of Contents**

- [**Real-Pod Dataset**](#real-pod-dataset)
  - [**Overview**](#overview)
  - [**Table of Contents**](#table-of-contents)
  - [Todo List](#todo-list)
  - [**Dataset Creation Process**](#dataset-creation-process)
    - [**1. Podcast Category**](#1-podcast-category)
    - [**2. Podcast Topics**](#2-podcast-topics)
    - [**3. Podcast Episode**](#3-podcast-episode)
    - [**Metadata**](#metadata)
  - [**Usage Guidelines**](#usage-guidelines)
    - [**Environment**](#environment)
    - [**Features**](#features)
    - [**File Outputs**](#file-outputs)
  - [**Ethical Considerations**](#ethical-considerations)
  - [**Citation**](#citation)


---
## Todo List

- [x] Add music separation control option
- [x] Add dynamic duration adjustment for dialogue selection

---

## **Dataset Creation Process**
To ensure the dataset is both diverse and representative of real-world podcasts, we followed a systematic approach to curate topics, select podcast episodes, and provide relevant metadata and processing tool.

### **1. Podcast Category**
We began by compiling a comprehensive list of podcast categories based on the taxonomy provided by `Apple Podcast` application. These 17 categories include:

- **Society & Culture**
- **Education**
- **Business**
- **Comedy**
- **Science & Technology**
- **Health & Fitness**
- **Arts**
- **News**
- **Sports**
- **History**
- **Kids & Family**
- **True Crime**
- **TV & Film**
- **Music**
- **Leisure**
- **Fiction**
- **Mental Health**

This list ensures broad coverage of podcast genres, reflecting the diversity of podcast content.

---

### **2. Podcast Topics**
To set up relevant and representative topics for each category:
1. **Initial Topic Suggestions**:
   - We used GPT-4 to generate **5 popular and distinct topics** per category. These topics reflected current trends and listener interests.

2. **Human Review and Refinement**:
   - Human annotators reviewed and refined the generated topics to ensure relevance, uniqueness, and alignment with real-world podcast themes.
   - **3 representative topics** were selected for each category, resulting in a final collection of 51 topics (17 categories × 3 topics).

---

### **3. Podcast Episode**
Once the topic collection was finalized, we manually searched and screened podcast episodes to identify those most relevant to the given topics. The selection process followed these principles:

**Selection Criteria**
1. **Topic Relevance**:
   - Episodes were chosen based on how well they align with the predefined topics.

2. **Multi-Speaker Dialogue**:
   - Preference was given to episodes featuring multiple speakers to facilitate evaluation of speaker diarization, dialogue segmentation, and related tasks.

3. **High Audio Quality**:
   - Episodes with clear and high-quality audio were prioritized to serve as benchmarks for audio quality.

4. **Diverse Audio Scenarios**:
   - Without Background Music: Episodes containing pure dialogue were included for evaluating speech clarity and voice quality.
   - With Background Music: Episodes with background music or sound effects were included to analyze the appropriateness and integration of such elements in podcast production.
---

### **Metadata**
The dataset includes a JSON file (`Podcast_51topics.json`) containing metadata for each podcast episode:
- **topic_id**: A unified topic id.
- **topic**: The specific topic (e.g., Online learning).
- **category**: The podcast category (e.g., Education, Comedy).
- **episode_link**: The publicly available URL where the episode can be accessed.
- **episode_title**: The title of the real podcast episode.

---

## **Usage Guidelines**

The dataset includes a processing tool (`data_process.py`) designed for downloading the curated real-world podcast dataset and preparing it for subsequent evaluation tasks. The pipeline can also be applied to data from podcast generation systems to be evaluated.

---

### **Environment**

See PodEval/README.md

---

### **Features**


#### **Command Line Arguments**
- `--mode`: Operation mode (`download`, `transcribe`, `dialog`, `summarize`, `full_process`)
- `--input`: Path to the input file/folder
- `--output`: Path to save output files
- `--min_dur`: Minimum duration of selected clips (seconds, default: 10)
- `--max_dur`: Maximum duration of selected clips (seconds, default: 30)
- `--num_dialogues`: Number of dialogues to select (default: 2)
- `--segment_duration`: Duration of each segment in seconds (default: 60)
- `--with_music`: Whether to separate music in the audio (optional)
- `--clear`: Remove intermediate files after processing (optional)

#### **1. Download Audio**
Download Real-Pod dataset (real-world podcast audio files) from the provided JSON file.

```bash
python data_process.py --mode download --input Podcast_51topics.json --output ./output
```

#### **2. Full Processing Pipeline**
- Do `Dialogue Extraction` and `Audio Summarization` for given dataset.

```bash
python data_process.py --mode full_process --input ./example/dataset --output ./example/processed --min_dur 10 --max_dur 30 --num_dialogues 2 --segment_duration 60 --with_music --clear
```

#### **3. Single Function Processing**

- **Transcription and Speaker Diarization:** Transcribes the audio into text using the Whisper modeland Performs speaker diarization to associate text segments with speakers using Pyannote.

```bash

python data_process.py --mode transcribe --input ./example/dataset --output ./example/processed --with_music
```

#### 
- **Dialogue Extraction:** Extract alternating dialogue segments with intelligent duration adjustment.

```bash
python data_process.py --mode dialog --input ./example/dataset --output ./example/processed --min_dur 10 --max_dur 30 --num_dialogues 2 --with_music
```


- **Audio Summarization:** Extract and merge three segments (start, middle, end) with high-quality beep sounds.


```bash
python data_process.py --mode summarize --input ./example/dataset --output ./example/processed/summary --segment_duration 60
```



---

### **File Outputs**

The processing pipeline generates various output files depending on the selected mode:

#### **Download**
- **Downloaded Files**: `.wav` files saved in the specified output directory.

#### **Transcription**
- **Vocals File**: `vocals.wav` - Separated vocals (if `--with_music` is used) or original audio
- **Diarized Transcription**: `diarized_transcription.txt` - Text with timestamps and speaker labels

#### **Dialogue Extraction**
- **Dialogue Clips**: `Clips_turns/` directory containing:
  - Extracted `.wav` files for alternating dialogues
  - `clips_info.json` - Metadata for all extracted clips
- **Selected Clips**: `Clips_turns_selected/` directory containing:
  - Selected dialogue clips based on duration criteria
  - `clips_info.json` - Metadata for selected clips only

#### **Summarization**
- **Summarized Audio**: `.wav` files with start, middle, and end segments merged with beep sounds

#### **Full Processing Mode**
- **Complete Output**: All above outputs organized in subdirectories by audio file name
- **Summary Files**: `summary.wav` for each processed audio file
- **Optional Cleanup**: Intermediate files removed if `--clear` is specified
- **Output Structure Example**

```
output_dir/
├── audio_file_1/
│   ├── vocals.wav
│   ├── diarized_transcription.txt
│   ├── summary.wav
│   ├── Clips_turns/
│   │   ├── dialog_1_speaker1-speaker2_10.5-25.3.wav
│   │   └── clips_info.json
│   └── Clips_turns_selected/
│       ├── dialog_1_speaker1-speaker2_10.5-25.3.wav
│       └── clips_info.json
└── audio_file_2/
    └── ...
```


---


## **Ethical Considerations**
**Responsible Use**: Users are encouraged to respect copyright laws and use the dataset for research and educational purposes only.

---

## **Citation**
If you use the Real-Pod dataset, please cite this project:
```
@dataset{PodEval,
}
```
# Objective Speech/Audio Evaluation Toolkit

This toolkit provides a comprehensive pipeline for evaluating the **objective quality metrics** of podcast or speech audio. It supports both file-by-file and module-by-module processing, with optional background music separation and reference-based evaluation. The toolkit is suitable for creators, researchers, and anyone interested in audio quality assessment.


---

## **Output Metrics Explained**


| Metric                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **DNSMOS(SIG)**         | Speech quality score (0-5, higher is better)                                 |
| **DNSMOS(BAK)**         | Background noise quality score (0-5, higher is better)                       |
| **DNSMOS(OVRL)**        | Overall speech quality score (0-5, higher is better)                         |
| **DNSMOS(P808_MOS)**    | Human-like MOS score (0-5, higher is better)                                 |
| **LOUD_IT**             | Integrated Loudness (LUFS, dB)                                               |
| **LOUD_TP**             | True Peak (dBTP)                                                             |
| **LOUD_LRA**            | Loudness Range (LRA, LU)                                                     |
| **LOUD_IT_SCORE**       | Score for integrated loudness (0-1, normalized)                              |
| **LOUD_TP_SCORE**       | Score for true peak (0-1, normalized)                                        |
| **LOUD_LRA_SCORE**      | Score for loudness range (0-1, normalized)                                   |
| **LOUD_AVG_SCORE**      | Average loudness score (0-1, normalized)                                     |
| **WER**                 | Word Error Rate (0-1, lower is better; requires reference text)              |
| **SPTD**                | Average timbre difference between speakers (0-1, higher = more different)    |
| **SIM**                 | Speaker similarity to reference (0-1, higher = more similar; needs reference)|
| **CASP**                | Speech-to-music harmony score (0-1, higher is better)                        |
| **SMR_MIN/MAX/AVG**     | Min/Max/Avg Speech-to-Music Ratio (dB) across segments                       |
| **SMR_BASIC_SCORE**     | Proportion of segments with SMR > 0 (0-1, higher is better)                  |
| **SMR_REF_SCORE_0.1_0.01** | Proportion of segments within reference SMR range (0-1)                   |

---


## **Workflow**

<p align="center">
  <img src="audio-objective-metrics-workflow.png" alt="Audio Objective Metrics Workflow" width="800"/>
</p>

---


## **Environment**

See PodEval/README.md

### More
- DNSMOS Models: The model can be downloaded from this [link](https://github.com/microsoft/DNS-Challenge). For convenient usage, we have uploaded the model files under `./DNSMOS`.
- Pyannote: Please follow the steps [here](https://github.com/pyannote/pyannote-audio) to create access tokens, and replace the `use_auth_token` in `./models.py` and `./Real_Pod/data_process.py`.

        ```python
        pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0", 
                use_auth_token="hf_xxx"
        )
        ```
---

## **Usage**


```bash
python main.py --input <input_audio_dir> --output <output_dir> [options]
```

### **Arguments**
- `--input` (required): Path to the directory containing `.wav` audio files
- `--output` (required): Directory to save results (per-file JSON and summary CSV)
- `--with_music`: If set, separates background music using Demucs
- `--mode`:  
  - `file` (default): **Processes one file at a time**. After each file is processed, intermediate files can be deleted immediately, saving disk space. This mode is ideal when disk space is limited.
  - `module`: **Processes all files through each module sequentially**. After finishing the computation of each metric for all files, the corresponding model is released from memory, reducing memory usage. This mode is suitable when memory resources are limited.
- `--ref_info`: Directory containing reference JSON files (for WER, speaker similarity, etc.), format instruction: [**Reference JSON Format**](#reference-json-format)
- `--segment_dir`: Directory containing provided segment files (optional)
  - If specified, the toolkit will use these provided segments instead of automatic segmentation. This is useful for models that support utterance-level (single-speaker segment) output, yielding more precise results compared to automatic segmentation. (E.g. `Example/segment_dir`)
- `--remove_itm`: If set, removes intermediate files after processing


### **Usage Scenarios**

The toolkit supports various evaluation scenarios based on your data characteristics. 
- MSE: music and sound effect
- No Ref Info: no ground truth transcripts
- No Segments: sentence-level speech segments are not provided.

#### **Scenario 1: With MSE + No Ref Info + No Segments**
- e.g. Real-Pod Dataset
- Expected Output: DNSMOS, LOUD_\*, SPTD, CASP, SMR_\*

```bash
python main.py \
  --input /path/to/real_pod_audio \
  --output /path/to/results \
  --with_music
```

#### **Scenario 2: With MSE + Ref Info + Provided Segments**
- e.g. PodAgent Data
- Expected Output: DNSMOS, LOUD_*, WER, SPTD, SIM, CASP, SMR_*

```bash
python main.py \
  --input /path/to/podagent_audio \
  --output /path/to/results \
  --with_music \
  --ref_info /path/to/reference_jsons \
  --segment_dir /path/to/provided_segments \
```


#### **Scenario 3:  No MSE + Ref Info + Provided Segments**
- e.g. MoonCast/MuyanTTS Data
- Expected Output: DNSMOS, LOUD_*, WER, SPTD, SIM

```bash
python main.py \
  --input /path/to/mooncast_audio \
  --output /path/to/results \
  --ref_info /path/to/reference_jsons \
  --segment_dir /path/to/provided_segments \
```


#### **Scenario 4: No MSE + Ref Info + No Segments**
- e.g. Dia/MOSS-TTSD Data
- Expected Output: DNSMOS, LOUD_*, WER, SPTD, SIM

```bash
python main.py \
  --input /path/to/dia_audio \
  --output /path/to/results \
  --ref_info /path/to/reference_jsons \
```

#### **Scenario 5: No MSE + No Ref Info + No Segments**
- e.g. NotebookLM Data
- Expected Output: DNSMOS, LOUD_*, SPTD

```bash
python main.py \
  --input /path/to/dia_audio \
  --output /path/to/results \
```

### **Scenario Comparison Table**

| Scenario | Background Music | Reference Info | Pre-segments | Expected Output |
|----------|------------------|----------------|--------------|-----------------|
| **Real-Pod** | ✅ Yes  | ❌ No | ❌ Auto | DNSMOS, LOUD_\*, SPTD, CASP, SMR_\* |
| **PodAgent** | ✅ Yes  | ✅ Yes | ✅ Yes | DNSMOS, LOUD_\*, WER, SPTD, SIM, CASP, SMR_\* |
| **MoonCast/MuyanTTS** | ❌ No | ✅ Yes | ✅ Yes | DNSMOS, LOUD_\*, WER, SPTD, SIM |
| **Dia/MOSS-TTSD** | ❌ No | ✅ Yes | ❌ Auto | DNSMOS, LOUD_\*, WER, SPTD, SIM |
| **NotebookLM** | ❌ No | ❌ No | ❌ Auto | DNSMOS, LOUD_*, SPTD |





---

## **Reference JSON Format**

Reference files should be in JSON format with the following fields (For a full example, see: `Example/ref_info_dir/Arts_19.json`):

- `role_mapping`:  
  A dictionary mapping each speaker role (as a string) to a reference audio and text.  
  Example:
  ```json
  "role_mapping": {
    "0": {
      "ref_audio": "path/to/speaker0_reference.wav",
      "ref_text": "Reference text for speaker 0"
    },
    "1": {
      "ref_audio": "path/to/speaker1_reference.wav",
      "ref_text": "Reference text for speaker 1"
    }
  }
  ```

- `dialogue`:  
  A list of utterances, each as a dictionary with:
  - `role`: The speaker role (string, matching keys in `role_mapping`)
  - `text`: The spoken content for that utterance  
  Example:
  ```json
  "dialogue": [
    {"role": "0", "text": "Hello, welcome to the show."},
    {"role": "1", "text": "Thank you, glad to be here."}
  ]
  ```

**Notes:**
- Each `role` in `dialogue` should correspond to a key in `role_mapping`.
- The `ref_audio` and `ref_text` in `role_mapping` provide reference samples for speaker similarity and WER evaluation.
- Both `role_mapping` and the `role`/`text` fields in `dialogue` are optional and can be omitted if not needed or cannot be provided for your evaluation scenario.

---

## **Acknowledgments**
- [**DNSMOS**](https://github.com/microsoft/DNS-Challenge): Speech quality evaluation models
- [**Demucs**](https://github.com/facebookresearch/demucs): Audio source separation
- [**Whisper**](https://github.com/openai/whisper): Automatic speech recognition
- [**Silero VAD**](https://github.com/snakers4/silero-vad): Voice activity detection
- [**Pyannote**](https://github.com/pyannote/pyannote-audio): Speaker diarization and embeddings
- [**DualBench/CASP**](https://github.com/wjtian-wonderful/DualBench): Speech/music harmony (CASP)
- [**pyloudnorm**](https://github.com/csteinmetz1/pyloudnorm): Loudness measurement
- [**HuggingFace Hub**](https://huggingface.co/): Model and checkpoint hosting

---

## **License**
See project root for license information. 
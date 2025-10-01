import os
import glob
import numpy as np
import pandas as pd
import librosa
import math
from tqdm import tqdm
import pyloudnorm as pyln
import torch
from models import get_casp_model
from utils import detect_speech_segments

def calculate_IDL_score(IDL, k1=0.0858 , k2=0.3291):

    if -17 <= IDL <= -14:
        return 1.0  # Ideal range
    if IDL < -17:
        return math.exp(-k1 * (-17 - IDL))
    if IDL > -14:
        return math.exp(-k2 * (IDL + 14))
    
def calculate_TP_score(TP, k3=4.605):

    if TP <= -1:
        return 1.0  # Ideal range
    else:
        return math.exp(-k3 * (TP + 1))  # Exponential decay for TP > -1 dBTP
    
def calculate_LRA_score(LRA, k4=1.1513, k5=0.2554):

    if 4 <= LRA <= 18:
        return 1.0  # Ideal range
    if LRA < 4:
        return math.exp(-k4 * (4 - LRA))
    if LRA > 18:
        return math.exp(-k5 * (LRA - 18))
    
def calculate_loudness(audio_file):
    """
    Calculate audio metrics: Integrated Loudness (LUFS), True Peak (dBTP), 
    and Loudness Range (LRA, LU). Scores are provided based on reference values.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        dict: A dictionary containing the calculated metrics and their scores.
    """
    # Load the audio file
    data, rate = librosa.load(audio_file, sr=None)

    # Create a loudness meter
    meter = pyln.Meter(rate)

    # Calculate Integrated Loudness (LUFS)
    integrated_loudness = meter.integrated_loudness(data)

    # Calculate True Peak (dBTP)
    oversample_factor = 4  # Increase resolution by 4x to find inter-sample peaks
    oversampled_data = librosa.resample(data, orig_sr=rate, target_sr=rate * oversample_factor)
    true_peak = np.max(np.abs(oversampled_data))  # Find the maximum absolute value
    true_peak_db = 20 * np.log10(true_peak)  # Convert to dBTP

    # Calculate Loudness Range (LRA)
    # Step 1: Calculate short-term loudness (3-second window)
    frame_size = int(rate * 3)  # 3-second frames
    hop_size = frame_size // 2  # 50% overlap
    short_term_loudness = []

    for i in range(0, len(data) - frame_size + 1, hop_size):
        frame = data[i:i + frame_size]
        loudness = meter.integrated_loudness(frame)
        short_term_loudness.append(loudness)

    # Step 2: Remove outliers (percentile-based)
    lower_bound = np.percentile(short_term_loudness, 10)  # 10th percentile
    upper_bound = np.percentile(short_term_loudness, 95)  # 95th percentile

    # Step 3: Compute LRA
    lra = upper_bound - lower_bound

    # Scoring rules
    # 1. Integrated Loudness Score
    #if -17 <= integrated_loudness <= -15:  # Within the range [-17, -15]
    #    score_loudness = 10  # Full score
    #else:
    #    score_loudness = max(0, 10 - abs(integrated_loudness - (-16)))  # Deduct points based on deviation
    score_loudness = calculate_IDL_score(integrated_loudness)

    # 2. True Peak Score
    #if true_peak_db <= -1:
    #    score_true_peak = 1  # Full score if True Peak is less than or equal to -1 dBTP
    #else:
    #    score_true_peak = max(0, 10 - (true_peak_db - (-1)) * 2)  # Deduct points for exceeding -1 dBTP
    score_true_peak = calculate_TP_score(true_peak_db)
    
    # 3. Loudness Range (LRA) Score
    #if 5 <= lra <= 10:
    #    score_lra = 10  # Perfect score if within the range [5, 10] LU
    #elif lra < 5:
    #    score_lra = max(0, 10 - (5 - lra))  # Deduct 1 point for every 1 LU below 5
    #else:
    #    score_lra = max(0, 10 - (lra - 10))  # Deduct 1 point for every 1 LU above 10
    score_lra = calculate_LRA_score(lra)

    # Calculate the overall score as the average of all three scores
    total_score = (score_loudness + score_true_peak + score_lra) / 3

    # Return a dictionary with the results
    return {
        "Integrated Loudness (LUFS)": float(integrated_loudness),
        "True Peak (dBTP)": float(true_peak_db),
        "Loudness Range (LRA, LU)": float(lra),
        "Scores": {
            "Loudness Score": float(score_loudness),
            "True Peak Score": float(score_true_peak),
            "LRA Score": float(score_lra),
            "Overall Score": float(total_score)
        }
    }


def calculate_casp_for_single_audio(vocals_path, no_vocals_path, model_path, duration_seg=10):
    """
    Split an audio segment into fixed-length chunks and pad as needed, returning the processed chunks and corresponding masks.

    Args:
        audio_tensor (Tensor): 1D tensor of audio data with shape (L,).
        target_len (int): Target length for each chunk.

    Returns:
        chunks (Tensor): shape (n_chunks, target_len)
        masks  (Tensor): shape (n_chunks, target_len)
    """
    
    def preprocess_audio_with_mask(audio_tensor, target_len=160000):
        length = audio_tensor.size(0)
        
        chunk_list = []
        mask_list = []
        for i in range(0, length, target_len):
            chunk = audio_tensor[i:i + target_len]
            chunk_len = chunk.size(0)
            mask = torch.ones(target_len)

            if chunk_len < target_len:
                # 不足目标长度时，重复该chunk直到满足长度
                repeat_times = (target_len // chunk_len) + 1
                repeated_chunk = chunk.repeat(repeat_times)[:target_len]
            else:
                repeated_chunk = chunk


            chunk_list.append(repeated_chunk.unsqueeze(0))  # shape: (1, target_len)
            mask_list.append(mask.unsqueeze(0))           # shape: (1, target_len)

        print(f"\tchunk_list: {len(chunk_list)}")
        chunks = torch.cat(chunk_list, dim=0)  # shape: (n_chunks, target_len)
        masks = torch.cat(mask_list, dim=0)    # shape: (n_chunks, target_len)

        return chunks, masks
    
    # Load audio files
    vocals_np, vocals_sr = librosa.load(vocals_path, sr=16000, mono=True)
    no_vocals_np, no_vocals_sr = librosa.load(no_vocals_path, sr=16000, mono=True)
    
    assert vocals_sr == no_vocals_sr, "Sampling rates of vocals and no-vocals files must match."
    assert len(vocals_np.shape) == 1, "vocals_np channel must be 1D."
    assert len(no_vocals_np.shape) == 1, "no_vocals_np channel must be 1D."

    # 3. 转换为 Tensor，归一化
    vocals_tensor = torch.tensor(vocals_np, dtype=torch.float32)
    no_vocals_tensor = torch.tensor(no_vocals_np, dtype=torch.float32)

    vocals_tensor = vocals_tensor / torch.max(torch.abs(vocals_tensor) + 1e-9)
    no_vocals_tensor = no_vocals_tensor / torch.max(torch.abs(no_vocals_tensor) + 1e-9)

    vocals, vocals_mask = preprocess_audio_with_mask(vocals_tensor, target_len=vocals_sr * duration_seg)
    no_vocals, no_vocals_mask = preprocess_audio_with_mask(no_vocals_tensor, target_len=no_vocals_sr * duration_seg)
    
    if len(vocals.shape) == 1:
        vocals = vocals.unsqueeze(0)
    if len(no_vocals.shape) == 1:
        no_vocals = no_vocals.unsqueeze(0)  
        

    '''
    ckpt_path = hf_hub_download(repo_id="wonderfuluuuuuuuuuuu/DualDub", filename="podcast-10s.ckpt")
    # ckpt_path = f"{model_path}/duration-5s.ckpt"
    pretrain_path = f"{model_path}/BEATs_iter3_plus_AS2M.pt"

    model = CASPWrapper(d_model=768, ckpt_path=pretrain_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval() 
    '''

    model = get_casp_model(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    vocals = vocals.to(device)
    no_vocals = no_vocals.to(device)
   
    '''
    logits_per_speech, logits_per_audio = model.model.inference_noscale(no_vocals, vocals, audio_mask=None, speech_mask=None)
    
    score = logits_per_speech
    score = torch.diag(score)

    score = (score.sum() / score.shape[0]).item()
    score = (score + 1) / 2

    summary = {
        "casp_score": score,
    }
    '''
    batch_size = 16  # Adjust batch size based on your GPU memory
    num_samples = vocals.shape[0]
    scores = []

    for i in range(0, num_samples, batch_size):
        v = vocals[i:i+batch_size].to(device)
        nv = no_vocals[i:i+batch_size].to(device)
        logits_per_speech, logits_per_audio = model.model.inference_noscale(nv, v, audio_mask=None, speech_mask=None)
        score = logits_per_speech
        score = torch.diag(score)
        scores.extend(score.tolist())
        del v, nv, logits_per_speech, logits_per_audio, score
        torch.cuda.empty_cache()

    casp_score = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "casp_score": (casp_score + 1) / 2,
    }

    return summary

def calculate_smr_score_human(smr, k1=0.6, k2=0.1155):
    """
    Calculate SMR score based on human auditory perception.

    Parameters:
        smr (float): Speech-to-Music Ratio (SMR) in LUFS.
        k1 (float): Decay rate for SMR < 5 (speech suppressed).
        k2 (float): Decay rate for SMR > 30 (background too low).

    Returns:
        float: The SMR score (range: 0 to 1).
    """
    if 5 <= smr <= 30:
        return 1.0  # Ideal range
    
    if smr < 0:
        return 0.0 # BMSE cannot be louder than Speech 

    if smr < 5:
        # Speech suppressed, fast decay
        return math.exp(-k1 * (5 - smr))
    
    if smr > 30:
        # Background too low, slower decay
        return math.exp(-k2 * (smr - 30))
        
def calculate_smr_for_speech_segments(vocals_path, no_vocals_path):
    """
    Calculate SMR (Speech-to-Music Ratio) for speech-active segments.
    
    Args:
        vocals_path (str): Path to the separated vocals audio file.
        no_vocals_path (str): Path to the separated no-vocals (background music) audio file.
        output_dir (str): Output directory for separated audio files.
    
    Returns:
        dict: A dictionary containing SMR values for each segment, overall statistics, 
              and a normalized score (0-10) based on the +10 to +20 dB reference range.
    """

    # Detect speech segments
    speech_timestamps = detect_speech_segments(vocals_path)
    
    # Load audio files
    vocals, sr = librosa.load(vocals_path, sr=16000)
    no_vocals, sr = librosa.load(no_vocals_path, sr=16000)

    # Create a loudness meter
    meter = pyln.Meter(sr)

    # Background music threshold (in dB, below which it's considered "no music")
    no_music_threshold = -60

    # Calculate SMR for each segment
    smr_values = []
    segments_within_basic_range = 0 # SMR>0
    segments_ref_score = 0 # ref: [6, 12]
    
    for speech_segment in speech_timestamps:
        # Skip segments shorter than 3 seconds
        if speech_segment['end'] - speech_segment['start'] < 3:
            continue
        
        # Convert overlap start and end times to sample indices
        start_sample = int(speech_segment['start'] * sr)
        end_sample = int(speech_segment['end'] * sr)

        # Extract the corresponding segments for vocals and background music
        vocals_segment = vocals[start_sample:end_sample]
        no_vocals_segment = no_vocals[start_sample:end_sample]

        # Calculate loudness 
        vocals_loudness = meter.integrated_loudness(vocals_segment)
        no_vocals_loudness = meter.integrated_loudness(no_vocals_segment)

        # Check if background music is detectable
        if no_vocals_loudness < no_music_threshold:
            # Segment is considered as "no background music", skip SMR evaluation
            continue

        # Calculate SMR (Speech-to-Music Ratio)
        smr = vocals_loudness - no_vocals_loudness
        smr_values.append(smr)

        # Check if the segment is within the reference range
        if smr > 0:
            segments_within_basic_range += 1
        #if smr >= reference_min and smr <= reference_max:
        #    segments_within_ref_range += 1
        segments_ref_score += calculate_smr_score_human(smr)

        #if reference_min <= smr <= reference_max:
        #    segments_within_range += 1
        #elif smr > reference_max: # vocals are louder than the reference range, not good but still acceptable
        #    segments_within_range += 0.5

    
    # Calculate the proportion of segments within the reference range
    basic_score = None
    ref_score = None
    valid_segments = len(smr_values)
    if valid_segments > 0:
        basic_score = segments_within_basic_range / valid_segments
        ref_score = segments_ref_score / valid_segments
    
    summary = {
        "smr_segments": valid_segments,
        "smr_avg": float(np.mean(smr_values)) if smr_values else None,
        "smr_max": float(max(smr_values)) if smr_values else None,
        "smr_min": float(min(smr_values)) if smr_values else None,
        "smr_basic_score": basic_score,
        "smr_ref_score": ref_score
    }
    return summary
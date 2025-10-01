import os
import glob
import numpy as np
import pandas as pd
import concurrent.futures
import soundfile as sf
import librosa
from tqdm import tqdm
from jiwer import wer
from scipy.spatial.distance import cosine
from models import get_onnx_sessions, get_whisper_model
from utils import extract_speaker_embeddings

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.chunk_seconds = 9.01
        self.sampling_rate = 16000
        self.primary_model_path = primary_model_path
        self.p808_model_path = p808_model_path
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, is_personalized_MOS):
        # Use cached ONNX sessions
        onnx_sess, p808_onnx_sess = get_onnx_sessions(self.primary_model_path, self.p808_model_path)
        # Initialize ONNX sessions in the worker
        #onnx_sess = ort.InferenceSession(self.primary_model_path)
        #p808_onnx_sess = ort.InferenceSession(self.p808_model_path)

        
        aud, input_fs = sf.read(fpath)
        #print("After sf.read, aud shape:", aud.shape)
        if len(aud.shape) > 1:
            aud = librosa.to_mono(aud.T) 
        fs = self.sampling_rate
        if input_fs != fs:
            audio = librosa.resample(y=aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(self.chunk_seconds*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - self.chunk_seconds)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+self.chunk_seconds)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            #print("Input feature shape:", input_features.shape)
            #print("P808 Input feature shape:", p808_input_features.shape)

            p808_mos = p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw,is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = float(np.mean(predicted_mos_ovr_seg_raw))
        clip_dict['SIG_raw'] = float(np.mean(predicted_mos_sig_seg_raw))
        clip_dict['BAK_raw'] = float(np.mean(predicted_mos_bak_seg_raw))
        clip_dict['OVRL'] = float(np.mean(predicted_mos_ovr_seg))
        clip_dict['SIG'] = float(np.mean(predicted_mos_sig_seg))
        clip_dict['BAK'] = float(np.mean(predicted_mos_bak_seg))
        clip_dict['P808_MOS'] = float(np.mean(predicted_p808_mos))
        return clip_dict


def calculate_dnsmos(process_dir, model_path, output_file = None):
    """
    Calculate speech quality scores for audio using DNSMOS.
    
    Parameters:
        process_dir: Directory containing input audio files.
        
    Returns:
        DNSMOS scores.
    """
    
    is_personalized_eval = False
    p808_model_path = f'{model_path}/DNSMOS/model_v8.onnx'
    clips = glob.glob(os.path.join(process_dir, "*.wav"))

    if is_personalized_eval:
        primary_model_path = f'{model_path}/pDNSMOS/sig_bak_ovr.onnx'
    else:
        primary_model_path = f'{model_path}/DNSMOS/sig_bak_ovr.onnx'
    compute_score = ComputeScore(primary_model_path, p808_model_path)
    
    # Process each audio file and calculate scores
    rows = []
    clip_count = 0
    total_scores = {
        "OVRL_raw": 0.0, "SIG_raw": 0.0, "BAK_raw": 0.0,
        "OVRL": 0.0, "SIG": 0.0, "BAK": 0.0, "P808_MOS": 0.0
    }
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        future_to_url = {executor.submit(compute_score, clip, is_personalized_eval): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data) 
                # Accumulate scores
                for key in total_scores:
                    total_scores[key] += data[key]
                clip_count += 1           

    # Calculate average scores
    if clip_count > 0:
        average_scores = {key: float(value / clip_count) for key, value in total_scores.items()}
    else:
        print("No valid audio files were processed.")
        return None
    
    # Save results to CSV if an output file is specified
    if output_file:
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    
    return average_scores

def calculate_dnsmos_for_single_audio(audio_file, model_path, output_file=None):
    """
    Calculate speech quality scores for a single audio file using DNSMOS.
    
    Parameters:
        audio_file (str): Path to the input audio file.
        model_path (str): Path to the directory containing model files.
        output_file (str, optional): Path to save the DNSMOS scores as a CSV file. If None, the result is printed.
        
    Returns:
        dict: DNSMOS scores for the audio file (if `output_file` is None).
    """
    # Define model paths
    p808_model_path = f'{model_path}/DNSMOS/model_v8.onnx'
    is_personalized_eval = False
    if is_personalized_eval:
        primary_model_path = f'{model_path}/pDNSMOS/sig_bak_ovr.onnx'
    else:
        primary_model_path = f'{model_path}/DNSMOS/sig_bak_ovr.onnx'

    # Initialize the ComputeScore class
    compute_score = ComputeScore(primary_model_path, p808_model_path)

    # Process the single audio file
    print(f"\tProcessing file: {audio_file}")
    try:
        result = compute_score(audio_file, is_personalized_eval)
    except Exception as exc:
        print(f"\tError processing {audio_file}: {exc}")
        return None

    # Save or return the result
    if output_file:
        df = pd.DataFrame([result])
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    else:
        print("DNSMOS Results:")
        print(result)
    return result

def analyze_timbre_differences(audio_dir):
    """
    Analyze the timbre differences between different speakers.

    Parameters:
        audio_dir (str): Directory containing the audio files.
    """

    embeddings = extract_speaker_embeddings(audio_dir)

    # average speaker embeddings
    speaker_means = {speaker: np.mean(vecs, axis=0) for speaker, vecs in embeddings.items()}

    # calculate cosine similarity
    speakers = list(speaker_means.keys())
    difference_scores = {}
    for i, speaker1 in enumerate(speakers):
        for speaker2 in speakers[i + 1:]:
            dist = cosine(speaker_means[speaker1], speaker_means[speaker2])/2  # divide by 2 to make the range 0-1
            difference_scores[(speaker1, speaker2)] = dist
    
    overall_diff = float(sum(difference_scores.values()) / len(difference_scores))
    return overall_diff

def calculate_speaker_similarity(audio_dir, ref_spk_dir, sent2refspk):
    """ 
    Calculate the similarity between generated and reference audio files.

    Parameters:
        audio_dir (str): Directory containing the segmented audio files.
        ref_spk_dir (str): Directory containing the reference audio files.
        sent2refspk (dict): Dictionary mapping sentence ID to reference speaker.
    """
    
    embeddings = extract_speaker_embeddings(ref_spk_dir)
    ref_spk_embs = {spk: np.mean(vecs, axis=0) for spk, vecs in embeddings.items()}
    
    embeddings = extract_speaker_embeddings(audio_dir)
    segment_embs = {seg_id: np.mean(vecs, axis=0) for seg_id, vecs in embeddings.items()}
    seg_spk_embs = {spk: [segment_embs[seg_id] for seg_id in segment_embs if sent2refspk[seg_id] == spk] for spk in ref_spk_embs.keys()}
    seg_spk_embs = {spk: np.mean(vecs, axis=0) for spk, vecs in seg_spk_embs.items()}
    
    similarity_scores = []
    for spk, emb in seg_spk_embs.items():
        ref_emb = ref_spk_embs[spk]
        similarity = 1 - cosine(emb, ref_emb)/2  # divide by 2 to make the range 0-1
        similarity_scores.append(similarity)
    
    return float(np.mean(similarity_scores))

def transcribe_audio(input_audio, output_dir, save = True):
    #whisper_model = whisper.load_model("base")
    whisper_model = get_whisper_model()
    
    whisper_result = whisper_model.transcribe(input_audio)

    if save:
        transcription_file = f"{output_dir}/transcription.txt"
        f = open(transcription_file, "w", encoding="utf-8")
    
    transcription = ""

    for segment in whisper_result["segments"]:
        # Each segment format: [start-end]: text
        if save:
            f.write(f"[{segment['start']:.2f}-{segment['end']:.2f}]: {segment['text']}\n")
        transcription += f"{segment['text']} "
    if save:
        print(f"    Transcription with timestamps saved to: {transcription_file}")
        f.close()

    return transcription
    
        
    
def calculate_wer(input_audio, reference_text, output_dir):

    transcription = transcribe_audio(input_audio, output_dir)
    
    if reference_text == "":
        return None

    error_rate = wer(reference_text, transcription)
    result = {"WER": error_rate}
    return result
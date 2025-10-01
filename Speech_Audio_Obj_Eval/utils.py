import os
import numpy as np
import librosa
import torch
from tqdm import tqdm
from pydub import AudioSegment
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from models import get_demucs_model, get_diarization_pipeline, get_vad_model, get_embedding_inference
from scipy.spatial.distance import cosine

def separate_audio_with_demucs(audio_file, output_dir="demucs_output"):
    """
    Use Demucs to separate audio.
    Parameters:
        audio_file: Path to the input audio file
        output_dir: Directory to save the separated files
    Returns:
        Paths to the separated vocals and accompaniment files
    """

    vocals_file = os.path.join(output_dir, "vocals.wav")
    accompaniment_file = os.path.join(output_dir, "accompaniment.wav")
    if os.path.exists(vocals_file) and os.path.exists(accompaniment_file):
        print(f"\tSeparated files already exist at: \n\t-{vocals_file}\n\t-{accompaniment_file}")
        return vocals_file, accompaniment_file
    
    os.makedirs(output_dir, exist_ok=True)
    #print(f"\tLoading Demucs model...")
    #demucs_model = get_model(name="htdemucs")  # load the Demucs model
    demucs_model = get_demucs_model()
   
    wav_data = AudioFile(audio_file).read(streams=0, samplerate=demucs_model.samplerate, channels=demucs_model.audio_channels)
    wav_data = wav_data.clone().detach().to("cuda" if torch.cuda.is_available() else "cpu").unsqueeze(0)
    
    # use the model to separate the audio
    print(f"\tSeparating audio...")
    sources = apply_model(demucs_model, wav_data, split=True)
    sources = sources.squeeze(0)

    # save the separated audio
    print(f"\tSaving separated audio...")
    accompaniment = None
    for source, name in zip(sources, demucs_model.sources):
        if name=='vocals':
            save_audio(source.cpu(), vocals_file, samplerate=demucs_model.samplerate) 
        else:
            if accompaniment is None:
                accompaniment = source.cpu() 
            else:
                accompaniment += source.cpu()

    if accompaniment is not None:
        save_audio(accompaniment, accompaniment_file, samplerate=demucs_model.samplerate)

    print(f"\tSaved to: \n\t-{vocals_file}\n\t-{accompaniment_file}")
    return vocals_file, accompaniment_file

def split_audio_by_speaker(audio_file, output_dir):
    """
    Split an audio file into multiple segments based on speaker diarization results.

    Parameters:
        audio_file (str): Path to the input audio file.
        output_dir (str): Directory where the segmented audio files will be saved.
    """

    if os.path.exists(output_dir):
        print(f"\tOutput directory already exists: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    '''
    print("\tLoading the speaker diarization model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0", 
        use_auth_token="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"
    )
    pipeline.to(torch.device("cuda"))
    '''
    pipeline = get_diarization_pipeline()

    print(f"\tProcessing the audio file: \n\t-{audio_file}")
    diarization = pipeline(audio_file)
    audio = AudioSegment.from_file(audio_file)

    # split the audio by speaker
    current_speaker = None
    current_start_time = 0
    current_end_time = 0
    save_index = 0
    for i, (turn, _, speaker) in tqdm(enumerate(diarization.itertracks(yield_label=True))):
        start_time = int(turn.start * 1000)  
        end_time = int(turn.end * 1000)      

        if speaker == current_speaker: # Same speaker
            current_end_time = end_time
            if current_end_time - current_start_time > 10000: # more than 10 seconds
                speaker_audio = audio[current_start_time:current_end_time]
                output_path = os.path.join(output_dir, f"segment_{save_index}_{current_speaker}.wav")
                speaker_audio.export(output_path, format="wav")
                save_index += 1

                current_speaker = None
                current_start_time = current_end_time
        else: # New speaker
            if current_speaker is not None and (current_end_time - current_start_time > 3000): # Save the previous speaker's segment
                speaker_audio = audio[current_start_time:current_end_time]
                output_path = os.path.join(output_dir, f"segment_{save_index}_{current_speaker}.wav")
                speaker_audio.export(output_path, format="wav")
                save_index += 1
            # Update current speaker
            current_speaker = speaker
            current_start_time = start_time
            current_end_time = end_time

    # Save the left audio segment
    if current_end_time - current_start_time > 3000:
        speaker_audio = audio[current_start_time:current_end_time]
        output_path = os.path.join(output_dir, f"segment_{save_index}_{current_speaker}.wav")
        speaker_audio.export(output_path, format="wav")

    print(f"\n\tAudio splitting complete! {save_index} segments have been saved to:\n\t-", output_dir)

def extract_speaker_embeddings(audio_dir):
    """
    Extract speaker embeddings for each audio file in the directory.

    Parameters:
        audio_dir (str): Directory containing the segmented audio files.
        model_name (str): Pretrained speaker embedding model to use.
    """
    
    #https://huggingface.co/pyannote/embedding
    inference = get_embedding_inference()

    embeddings = {}
    for file_name in tqdm(os.listdir(audio_dir)):
        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(audio_dir, file_name)
        audio_embedding = inference(file_path)
        # `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

        # format: segment_Y_SPEAKER_X.wav
        speaker = file_name.split("_")[-1].split(".")[0]  # Extract "SPEAKER_X"
        if speaker not in embeddings:
            embeddings[speaker] = []
        embeddings[speaker].append(audio_embedding)

    return embeddings

def detect_speech_segments(audio_path, min_silence_duration=0.5):
    """
    Detect speech activity segments using Silero VAD.
    Args:
        audio_path (str): Path to the input audio file.
        sampling_rate (int): Sampling rate (default is 16kHz).

    Returns:
        list: A list of speech activity segments, where each segment is a dictionary 
              {start, end} (in milliseconds).
    """
    
    # Load VAD model
    #model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    model, utils = get_vad_model()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Load the audio file and detect its original sampling rate
    wav, original_sr = librosa.load(audio_path, sr=None)

    # Resample the audio to 16kHz if necessary (Silero VAD requires 16kHz)
    target_sr = 16000
    if original_sr != target_sr:
        wav = librosa.resample(wav, orig_sr=original_sr, target_sr=target_sr)

    # Detect speech activity
    wav = torch.from_numpy(wav).unsqueeze(0)
    speech_timestamps = get_speech_timestamps(wav, model)

    # Merge segments with short gaps (e.g., < 1s)
    merge_threshold = int(min_silence_duration * target_sr)
    merged_segments = []
    for segment in speech_timestamps:
        if not merged_segments or segment['start'] - merged_segments[-1]['end'] > merge_threshold:
            merged_segments.append(segment)
        else:
            merged_segments[-1]['end'] = segment['end']

    # Convert merged_segments from samples to seconds
    for segment in merged_segments:
        segment['start'] = segment['start'] / target_sr
        segment['end'] = segment['end'] / target_sr

    return merged_segments

def detect_audio_activity(audio_path, threshold=0.01, min_silence_duration=0.5, sr=16000):
    """
    Detect non-silent parts of an audio file based on amplitude threshold.
    
    Args:
        audio_path (str): Path to the audio file.
        threshold (float): Amplitude threshold to detect sound (default: 0.01).
        min_silence_duration (float): Minimum silence duration (in seconds) to separate segments.
        sr (int): Target sampling rate for audio (default: 16000 Hz).

    Returns:
        list: A list of dictionaries with start and end times (in seconds) of sound segments.
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sr)
    
    # Calculate the amplitude (absolute value)
    amplitude = np.abs(audio)
    
    # Identify frames where the amplitude exceeds the threshold
    non_silent_indices = np.where(amplitude > threshold)[0]
    
    if len(non_silent_indices) == 0:
        return []  # No sound detected
    
    # Convert minimum silence duration to samples
    min_silence_samples = int(min_silence_duration * sr)
    
    # Group contiguous non-silent indices into segments
    segments = []
    start = non_silent_indices[0]
    for i in range(1, len(non_silent_indices)):
        if non_silent_indices[i] - non_silent_indices[i - 1] > min_silence_samples:
            # New segment
            end = non_silent_indices[i - 1]
            segments.append({"start": start / sr, "end": end / sr})
            start = non_silent_indices[i]
    # Add the last segment
    segments.append({"start": start / sr, "end": non_silent_indices[-1] / sr})
    
    return segments

def cluster_by_refspk(segment_dir, ref_spk_dir):
    """
    Cluster segments by reference speakers and rename files accordingly.
    
    Parameters:
        segment_dir (str): Directory containing the segmented audio files.
        ref_spk_dir (str): Directory containing the reference speaker audio files.
    """
    print(f"\tClustering segments by reference speakers...")
    
    # Get embedding inference model
    inference = get_embedding_inference()
    
    # Step 1: Extract reference speaker embeddings
    print(f"\tExtracting reference speaker embeddings from {ref_spk_dir}...")
    ref_spk_embeddings = {}
    for file_name in os.listdir(ref_spk_dir):
        if not file_name.endswith(".wav"):
            continue
            
        file_path = os.path.join(ref_spk_dir, file_name)
        try:
            # Extract speaker name from filename (assuming format like "speaker_name.wav")
            speaker_name = file_name.split(".")[0].split("_")[-1]
            embedding = inference(file_path)
            ref_spk_embeddings[speaker_name] = embedding
            print(f"\t\tExtracted embedding for reference speaker: {speaker_name}")
        except Exception as e:
            print(f"\t\tError extracting embedding from {file_name}: {e}")
            continue
    
    if not ref_spk_embeddings:
        print(f"\tNo valid reference speaker embeddings found in {ref_spk_dir}")
        return
    
    # Step 2: Extract embeddings from segments and find closest reference speaker
    print(f"\tProcessing segments in {segment_dir}...")
    segment_files = [f for f in os.listdir(segment_dir) if f.endswith(".wav")]
    
    for file_name in tqdm(segment_files, desc="Processing segments"):
        file_path = os.path.join(segment_dir, file_name)
        
        try:
            # Extract embedding from segment
            segment_embedding = inference(file_path)
            
            # Find closest reference speaker
            best_speaker = None
            best_similarity = -1
            
            for ref_speaker, ref_embedding in ref_spk_embeddings.items():
                # Calculate cosine similarity (1 - distance, so higher is more similar)
                similarity = 1 - cosine(segment_embedding, ref_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = ref_speaker
            
            if best_speaker is not None:
                # Parse the original filename to extract save_index
                # Expected format: segment_{save_index}_{current_speaker}.wav
                parts = file_name.split("_")
                if len(parts) >= 3:
                    save_index = parts[1]
                    new_filename = f"segment_{save_index}_SPEAKER_{best_speaker}.wav"
                    new_file_path = os.path.join(segment_dir, new_filename)
                    
                    # Rename the file
                    if file_path != new_file_path:
                        os.rename(file_path, new_file_path)
                        #print(f"\t\tRenamed {file_name} -> {new_filename} (similarity: {best_similarity:.3f})")
                    else:
                        print(f"\t\tFile {file_name} already has correct speaker name")
                else:
                    print(f"\t\tWarning: Could not parse filename format for {file_name}")
            else:
                print(f"\t\tWarning: Could not find matching reference speaker for {file_name}")
                
        except Exception as e:
            print(f"\t\tError processing {file_name}: {e}")
            continue
    
    print(f"\tSpeaker clustering complete!")

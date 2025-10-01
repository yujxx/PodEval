import os
import json
import random
import shutil
import torch
import whisper
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from pydub.generators import Sine
from pyannote.audio import Pipeline
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Podcast Processing Tool: Download, transcribe, extract dialogues, and summarize audio."
    )

    # Mode argument to specify the operation
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["download", "transcribe", "dialog", "summarize", "full_process"],
                       help="Operation mode: download, transcribe, dialog, summarize, or full_process")

    # Common arguments
    parser.add_argument("--input", type=str, required=True, help="Path to the input file/folder.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output files.")

    # Dialog-related arguments (used by dialog and full_process modes)
    parser.add_argument("--min_dur", type=int, default=10, help="Minimum duration of selected clips (seconds).")
    parser.add_argument("--max_dur", type=int, default=30, help="Maximum duration of selected clips (seconds).")
    parser.add_argument("--num_dialogues", type=int, default=2, help="Number of dialogues to select.")

    # Summarization-related arguments (used by summarize and full_process modes)
    parser.add_argument("--segment_duration", type=int, default=60, help="Duration of each segment in seconds.")

    # Processing-related arguments
    parser.add_argument("--with_music", action="store_true", help="Whether to separate music in the audio.")
    parser.add_argument("--clear", action="store_true", help="Remove intermediate files after processing.")

    return parser.parse_args()

def get_title(url):
    """
    Retrieve the title of a YouTube video (or other supported media) using yt-dlp without downloading the video.

    Args:
        url (str): The URL of the video for which the title needs to be fetched.

    Returns:
        str: The title of the video as a string.
    """
    result = subprocess.run(
        [
            "yt-dlp",
            "--print", "%(title)s",  # Fetch the title without downloading the video
            url
        ],
        capture_output=True,
        text=True,
        check=True
    )
    title = result.stdout.strip()
    return title

def download_by_json(json_file, output_dir="downloads"):
    """
    Download audio files from links in a JSON file, update the JSON with additional metadata, 
    and save the updated JSON to a new file.

    Args:
        json_file (str): Path to the input JSON file containing records with download links.
        output_dir (str, optional): Directory to save the downloaded audio files. Default is "downloads".

    Returns:
        None
    """

    output_dir = f"{output_dir}/downloaded_podcasts"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_file) as f:
        data = json.load(f)
    
    for item in tqdm(data):
        try:
            # Download audio as a WAV file using yt-dlp
            subprocess.run(
                [
                    "yt-dlp",
                    "--extract-audio", 
                    "--audio-format", "wav",
                    "--output", f"{output_dir}/{item['category']}_{item['topic_id']}.wav",
                    item["episode_link"]
                ],
                check=True
            )
        
        except subprocess.CalledProcessError as e:
            print(f"    Download failed for: {item['episode_link']}, error: {e}")
            continue
    
    return output_dir

def seperate_vocals(input_audio, output_dir):
    demucs_model = pretrained.get_model("htdemucs")
    os.makedirs(output_dir, exist_ok=True)

    # Load the audio file
    print(f"    Loading audio file: {input_audio}")
    wav_data = AudioFile(input_audio).read(
        streams=0, 
        samplerate=demucs_model.samplerate, 
        channels=demucs_model.audio_channels
    )
    wav_data = torch.tensor(
        wav_data, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).unsqueeze(0)

    # Apply Demucs model to separate sources
    print("    Separating sources (may take a while if the audio is long)...")
    sources = apply_model(demucs_model, wav_data, split=True)

    # Save the separated sources
    vocals_audio = os.path.join(output_dir, "vocals.wav")
    sources = sources.squeeze(0)
    for source, name in zip(sources, demucs_model.sources):
        if name == 'vocals':
            save_audio(source.cpu(), vocals_audio, samplerate=demucs_model.samplerate)
            print(f"    Vocals file saved to: {vocals_audio}")

    return vocals_audio
    
def transcribe_audio(input_audio, output_dir):
    whisper_model = whisper.load_model("base")
    whisper_result = whisper_model.transcribe(input_audio)

    transcription_file = f"{output_dir}/transcription.txt"
    with open(transcription_file, "w", encoding="utf-8") as f:
        for segment in whisper_result["segments"]:
            # Each segment format: [start-end]: text
            f.write(f"[{segment['start']:.2f}-{segment['end']:.2f}]: {segment['text']}\n")
    
    print(f"\t\tTranscription with timestamps saved to: {transcription_file}")
    return transcription_file, whisper_result
    
def perform_diarization(input_audio):
    pyannote_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token="hf_xxx",  # Replace with your Hugging Face token
    )
    pyannote_pipeline.to(torch.device("cuda"))
    diarization_result = pyannote_pipeline(input_audio)

    return diarization_result
    
def podcast_to_text(input_audio, output_dir, with_music=False):
    """
    Convert a podcast audio file into text with speaker diarization and timestamps.
    
    This function performs three main steps:
    1. Separates vocals from background music using the Demucs model.
    2. Transcribes the separated vocals into text with timestamps using the Whisper model.
    3. Performs speaker diarization to attribute text segments to speakers using Pyannote.audio.

    Args:
        input_audio (str): Path to the input podcast audio file.
        output_dir (str): Directory where output files (vocals, transcriptions, etc.) will be saved.
        with_music (bool): Whether to separate music in the audio. If False, copies original audio.
    
    Returns:
        None: Outputs are saved as files in the specified directory.
    """

    vocals_audio = os.path.join(output_dir, "vocals.wav")
    if not os.path.exists(vocals_audio):
        if with_music:
            print("\t\t[podcast_to_text] Separating vocals and background music...")
            vocals_audio = seperate_vocals(input_audio, output_dir)
        else:
            print("\t\t[podcast_to_text] No need to seperate vocals and background music, copy the original audio file...")
            shutil.copyfile(input_audio, vocals_audio)
    else:
        print(f"    Vocals file already exists: {vocals_audio}")

    print("\t\t[podcast_to_text] Transcribing vocals using Whisper...")
    transcription_file = f"{output_dir}/transcription.txt"
    whisper_result = None
    if not os.path.exists(transcription_file):
        transcription_file, whisper_result = transcribe_audio(vocals_audio, output_dir)
    else:
        print(f"    Transcription file already exists: {transcription_file}")
    
    print("\t\t[podcast_to_text] Performing speaker diarization...")
    diarized_transcription_file = f"{output_dir}/diarized_transcription.txt"
    if not os.path.exists(diarized_transcription_file):
        
        diarization_result = perform_diarization(vocals_audio)

        # Load Whisper transcription segments if not already loaded
        if whisper_result is None:
            if not os.path.exists(transcription_file):
                raise FileNotFoundError(f"Transcription file {transcription_file} does not exist!")
            
            whisper_segments = []
            with open(transcription_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        timestamp, text = line.split("]: ", 1)
                        start, end = map(float, timestamp.strip("[]").split("-"))
                        whisper_segments.append({"start": start, "end": end, "text": text})
        else:
            whisper_segments = whisper_result["segments"]

        # Save diarized transcription
        with open(diarized_transcription_file, "w", encoding="utf-8") as f:
            for segment in whisper_segments:
                segment_duration = segment["end"] - segment["start"]
                speaker_durations = {}

                for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                    overlap_start = max(turn.start, segment["start"])
                    overlap_end = min(turn.end, segment["end"])
                    if overlap_start < overlap_end:
                        overlap_duration = overlap_end - overlap_start
                        if speaker not in speaker_durations:
                            speaker_durations[speaker] = 0
                        speaker_durations[speaker] += overlap_duration

                # Identify the dominant speaker for the segment
                if speaker_durations:
                    dominant_speaker = max(speaker_durations, key=speaker_durations.get)
                    f.write(
                        f"[Speaker {dominant_speaker}] [{segment['start']:.2f}-{segment['end']:.2f}]: {segment['text']}\n"
                    )

        print(f"\t\t[podcast_to_text] Diarized transcription saved to: {diarized_transcription_file}")

def merge_segments(segments):
    """Merge consecutive speech segments into a single segment."""
    if not segments:
        return {}
    
    return {
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": " ".join(seg["text"] for seg in segments)
    }

def get_alternating_groups(segments, num_segments=-1):
    """
    Extract alternating dialog segments from the list of segments.
    Args:
        segments (List[Dict[str, Any]]): List of segments with speaker and timestamp information.
        num_segments (int): Maximum number of alternating dialog segments to extract. default is -1 (all segments).
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing alternating dialog segments.
    """
    alternating_groups = []
    i = 0
    while i < len(segments)-1:
        # Find speaker change
        while i < len(segments)-1 and segments[i]["speaker"] == segments[i+1]["speaker"]:
            i += 1
        
        if i < len(segments)-1:
            current_speaker = segments[i]["speaker"]
            next_speaker = segments[i+1]["speaker"]
            
            # Collect the last 2 segments of the current speaker
            current_segments = []
            j = i
            end = segments[j]["end"]
            while j >= 0 and segments[j]["speaker"] == current_speaker:
                current_segments.insert(0, segments[j])
                start = segments[j]["start"]
                if end - start > 10 or len(current_segments) > 2:
                    break
                j -= 1
            
            # Collect the first 2 segments of the next speaker
            next_segments = []
            j = i + 1
            start = segments[j]["start"]
            while j < len(segments) and segments[j]["speaker"] == next_speaker and len(next_segments) < 2:
                next_segments.append(segments[j])
                end = segments[j]["end"]
                if end - start > 10:
                    break
                j += 1
                
            # If a full alternating dialog is found
            if current_segments and next_segments:
                group = {
                    "start": current_segments[0]["start"],
                    "end": next_segments[-1]["end"],
                    "first_speaker_segments": current_segments,
                    "second_speaker_segments": next_segments
                }
                alternating_groups.append(group)
                
                # Stop if enough segments are found
                if num_segments>0 and len(alternating_groups) >= num_segments:
                    break
            
            i = i + 1
        else:
            break
    
    return alternating_groups

def process_dialog_groups(alternating_groups, audio_file, output_dir):
    """
    Processes alternating dialog groups, extracts audio clips, and generates clip metadata.

    Args:
        alternating_groups (List[Dict[str, Any]]): List of alternating dialog groups. 
            Each group contains:
            - "start" (float): Start time of the group in seconds.
            - "end" (float): End time of the group in seconds.
            - "first_speaker_segments" (list): Segments of the first speaker.
            - "second_speaker_segments" (list): Segments of the second speaker.
        audio_file (str): Path to the input audio file.
        output_dir (str): Directory to save the generated audio clips.

    Returns:
        List[Dict[str, Any]]: List of metadata dictionaries for each processed clip.
            Each dictionary contains:
            - "dialog_id" (int): The ID of the dialog clip.
            - "audio_file" (str): The filename of the saved audio clip.
            - "time_range" (dict): Start and end times of the clip.
            - "first_speaker" (str): The first speaker in the dialog.
            - "second_speaker" (str): The second speaker in the dialog.
            - "text" (str): Merged dialog text from both speakers.
    """
    try:
        audio = AudioSegment.from_file(audio_file)
        # Convert to 16kHz mono
        audio = audio.set_frame_rate(16000).set_channels(1)
    except Exception as e:
        print(f"    Error loading audio file: {e}")
        return []
    
    clips_info = {
        "audio_file": '/'.join(audio_file.split('/')[-4:]),
        "total_clips": len(alternating_groups),
        "clips": []
    }

    os.makedirs(output_dir, exist_ok=True)
    for i, group in enumerate(alternating_groups, 1):
        try:
            # Extract the audio clip
            start_ms = int(group["start"] * 1000)
            end_ms = int(group["end"] * 1000)

            clip = audio[start_ms:end_ms]
            
            # Generate a unique filename for the audio clip
            first_speaker = group["first_speaker_segments"][0]["speaker"]
            second_speaker = group["second_speaker_segments"][0]["speaker"]
            time_range = f"{group['start']:.2f}-{group['end']:.2f}"
            audio_filename = f"dialog_{i}_{first_speaker}-{second_speaker}_{time_range}.wav"
            
            audio_path = os.path.join(output_dir, audio_filename)
            
            # Save the audio clip
            clip.export(audio_path, format="wav")
            
            # Merge segments for both speakers
            first_speaker_merged = merge_segments(group["first_speaker_segments"])
            second_speaker_merged = merge_segments(group["second_speaker_segments"])
            
            # Create a metadata dictionary for the clip
            clip_info = {
                "dialog_id": i,
                "audio_file": audio_filename,
                "time_range": {
                    "start": group["start"],
                    "end": group["end"]
                },
                "first_speaker": first_speaker,
                "second_speaker": second_speaker,
                "text": first_speaker_merged["text"] + " " + second_speaker_merged["text"]
            }
            
            clips_info["clips"].append(clip_info)
                
        except Exception as e:
            print(f"    Error in processing segment {i}: {e}")
            continue

    return clips_info

def extract_dialog_segments(audio_file, diarized_trans_file, output_dir, num_segments=-1):
    """
    Extract alternating dialog segments from a transcription file, save corresponding audio clips,
    and generate metadata for each dialog segment.

    Args:
        audio_file (str): Path to the input audio file.
        diarized_trans_file (str): Path to the transcription file with speaker and timestamp information.
        output_dir (str): Directory to store the extracted audio clips and metadata.
        num_segments (int, optional): Maximum number of alternating dialog segments to process. Default is -1 (all segments).

    Returns:
        dict: A dictionary containing metadata about the processed audio clips.
    """
    
    print("\t\t[extract_dialog_segments] Loading diarized_transcription.txt...")
    if not os.path.exists(diarized_trans_file):
        podcast_to_text(audio_file, output_dir)
        diarized_trans_file = f"{output_dir}/diarized_transcription.txt"
    
    print("\t\t[extract_dialog_segments] Parse each line and store segment information...")
    segments = []
    with open(diarized_trans_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip() or not line.startswith('[Speaker'):
                continue
                
            # Parse the line to extract speaker, start time, end time, and text.
            speaker_part, time_text = line.split("] [")
            speaker = speaker_part.replace("[Speaker ", "")
            time_part, text = time_text.split("]:")
            start, end = map(float, time_part.split("-"))
            
            segments.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text.strip()
            })
    
    print("\t\t[extract_dialog_segments] Extracting alternating dialog segments...")
    alternating_groups = get_alternating_groups(segments, num_segments)

    print("\t\t[extract_dialog_segments] Post-processing dialog segments...")
    all_dialog_dir = os.path.join(output_dir, "Clips_turns")
    clips_info = process_dialog_groups(
        alternating_groups=alternating_groups,
        audio_file=audio_file,
        output_dir=all_dialog_dir
    )
    
    # Save the metadata to a JSON file
    json_path = os.path.join(all_dialog_dir, 'clips_info.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clips_info, f, ensure_ascii=False, indent=2)
    
    print(f"\t\t[extract_dialog_segments] Processed [{len(clips_info['clips'])}] segments and saved in: {json_path}")

    return json_path
    

def clear_directory(directory: str):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"\t\t[clear_directory] Cannot delete file {file_path}: {e}")

def select_and_copy_dialogues(clip_info_file, output_dir, min_dur = 10, max_dur = 30, num_dialogues = 2):
    """
    Selects dialog clips based on duration, copies their audio files to the output directory,
    and saves metadata for the selected clips.

    Args:
        clip_info_file (str): Path to the JSON file containing dialog clip metadata.
        output_dir (str): Directory to save the selected dialog clips and metadata.
        num_dialogues (int, optional): Number of dialog clips to select. Default is 2.

    Returns:
        None
    """ 

    os.makedirs(output_dir, exist_ok=True)
    
    with open(clip_info_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clips = data["clips"]
    if len(clips) < num_dialogues:
        print(f"    Dialogue number [{len(clips)}] less than {num_dialogues}.")
        num_dialogues = len(clips)
    
    # Filter clips by duration (10 to 30 seconds)
    filtered_clips = [
        clip for clip in clips
        if min_dur <= (clip["time_range"]["end"] - clip["time_range"]["start"]) <= max_dur
    ]
    
    # If not enough clips, expand the duration range
    while len(filtered_clips) < num_dialogues and min_dur > 5 and max_dur < 35:
        min_dur -= 1
        max_dur += 1
        print(f"    New duration range: {min_dur} - {max_dur} seconds")
        filtered_clips = [
            clip for clip in clips
            if min_dur <= (clip["time_range"]["end"] - clip["time_range"]["start"]) <= max_dur
        ]

    if not filtered_clips:
        print(" No dialog clips found.")
        return
    
    selected_clips = random.sample(filtered_clips, num_dialogues)
    
    selected_data = {
        "audio_file": data.get("audio_file", "unknown"),
        "selected_clips": selected_clips
    }

    # Delete existing files in the output directory
    clear_directory(output_dir)
    
    # Copy selected audio files to the output directory
    for clip in selected_clips:
        audio_file = clip["audio_file"]
        source_path = f"{os.path.dirname(clip_info_file)}/{audio_file}"
        dest_path = os.path.join(output_dir, audio_file)
        
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print(f"    Not found: {source_path}")
    
    output_json_file = os.path.join(output_dir, "clips_info.json")
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=2)
    
    print(f"    Saved to: {output_json_file}")


def summarize_audio_with_segments(audio_file, output_file, segment_duration = 60):
    """
    Extracts three segments (start, middle, end) from an audio file, adds a beep sound between them,
    and merges them into a single audio file.

    Args:
        audio_file (str): Path to the input audio file.
        output_file (str): Path to save the merged output audio.
        segment_duration (int, optional): Duration of each extracted segment in seconds. Default is 60 seconds.

    Returns:
        None
    """
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)
    
    duration_ms = len(audio)
    segment_ms = segment_duration * 1000
    
    start_segment = audio[:segment_ms]
    mid_start = (duration_ms - segment_ms) // 2
    middle_segment = audio[mid_start:mid_start+segment_ms]
    end_segment = audio[-segment_ms:]
    
    # generate beep sound with 16kHz mono
    beep = Sine(1000).to_audio_segment(duration=300)
    beep = beep.set_frame_rate(16000).set_channels(1)  # convert to 16kHz mono
    beep = beep - 10  # reduce volume
    
    # fade out the segments
    fade_duration = 2000  # 2 seconds fade out
    start_segment = start_segment.fade_out(fade_duration)
    middle_segment = middle_segment.fade_out(fade_duration)
    
    # merge segments with beep
    merged = start_segment + beep + middle_segment + beep + end_segment
    
    # save
    merged.export(output_file, format="wav")
    
    print(f"    Save to: {output_file}")

def process_all(args):
    """
    Run all steps sequentially for multiple audio files.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    
    downloaded_files = os.listdir(args.input)
    for audio_file in tqdm(downloaded_files, desc="Processing audio files"):
        if audio_file.endswith(".wav"):
            print(f"Processing {audio_file}...")
            output_dir = os.path.join(args.output, f"{os.path.splitext(audio_file)[0]}")
            os.makedirs(output_dir, exist_ok=True)

            print("\tStep 1: Transcribing and diarizing audio files...")
            input_audio = os.path.join(args.input, audio_file)
            podcast_to_text(input_audio, output_dir, args.with_music)

            print("\tStep 2: Extracting and select dialog segments...")
            vocals_file = os.path.join(output_dir, "vocals.wav")
            transcription_file = os.path.join(output_dir, "diarized_transcription.txt")
            if os.path.exists(vocals_file) and os.path.exists(transcription_file):
                clips_output_dir = os.path.join(output_dir, "Clips_turns")
                selected_clips_dir = os.path.join(output_dir, "Clips_turns_selected")
                if not os.path.exists(selected_clips_dir):
                    clip_info_file = extract_dialog_segments(
                        audio_file=vocals_file,
                        diarized_trans_file=transcription_file,
                        output_dir=clips_output_dir,
                    )

                    select_and_copy_dialogues(
                        clip_info_file=clip_info_file,
                        output_dir=selected_clips_dir,
                        min_dur=args.min_dur,
                        max_dur=args.max_dur,
                        num_dialogues=args.num_dialogues
                    )
                else:
                    print(f"    Selected clips already exist in: {selected_clips_dir}. Skip.")
            else:
                print(f"    Vocals or transcription file not found: {vocals_file} or {transcription_file}")

            print("\tStep 3: Summarizing audio files...")
            output_file = f"{output_dir}/summary.wav" 
            summarize_audio_with_segments(
                audio_file=input_audio,
                output_file=output_file,
                segment_duration=args.segment_duration
            )

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_arguments()

    # Execute the appropriate mode
    if args.mode == "download":
        _ = download_by_json(args.input, args.output)
    
    elif args.mode == "transcribe":
        audio_files = os.listdir(args.input)
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            if audio_file.endswith(".wav"):
                print(f"Processing {audio_file}...")
                output_dir = os.path.join(args.output, f"{os.path.splitext(audio_file)[0]}")
                os.makedirs(output_dir, exist_ok=True)

                input_audio = os.path.join(args.input, audio_file)
                podcast_to_text(input_audio, output_dir, args.with_music)
    
    elif args.mode == "dialog":
        audio_files = os.listdir(args.input)
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            if audio_file.endswith(".wav"):
                print(f"Processing {audio_file}...")
                output_dir = os.path.join(args.output, f"{os.path.splitext(audio_file)[0]}")
                os.makedirs(output_dir, exist_ok=True)

                print("\tStep 1: Transcribing and diarizing audio files...")
                input_audio = os.path.join(args.input, audio_file)
                podcast_to_text(input_audio, output_dir, args.with_music)

                print("\tStep 2: Extracting and select dialog segments...")
                vocals_file = os.path.join(output_dir, "vocals.wav")
                transcription_file = os.path.join(output_dir, "diarized_transcription.txt")
                if os.path.exists(vocals_file) and os.path.exists(transcription_file):
                    clips_output_dir = os.path.join(output_dir, "Clips_turns")
                    selected_clips_dir = os.path.join(output_dir, "Clips_turns_selected")
                    if not os.path.exists(selected_clips_dir):
                        clip_info_file = extract_dialog_segments(
                            audio_file=vocals_file,
                            diarized_trans_file=transcription_file,
                            output_dir=clips_output_dir,
                        )

                        select_and_copy_dialogues(
                            clip_info_file=clip_info_file,
                            output_dir=selected_clips_dir,
                            min_dur=args.min_dur,
                            max_dur=args.max_dur,
                            num_dialogues=args.num_dialogues
                        )
                    else:
                        print(f"    Selected clips already exist in: {selected_clips_dir}. Skip.")
                else:
                    print(f"    Vocals or transcription file not found: {vocals_file} or {transcription_file}")
    
    elif args.mode == "summarize":
        audio_files = os.listdir(args.input)
        for audio_file in tqdm(audio_files, desc="Summarizing audio files..."):
            if audio_file.endswith(".wav"):
                print(f"Processing {audio_file}...")
                output_file = os.path.join(args.output, f"{os.path.splitext(audio_file)[0]}.wav")
                os.makedirs(args.output, exist_ok=True)

                input_audio = os.path.join(args.input, audio_file)
                summarize_audio_with_segments(
                    audio_file=input_audio,
                    output_file=output_file,
                    segment_duration=args.segment_duration
                )
    
    elif args.mode == "full_process":
        process_all(args)

    if args.clear:
        for subdir in tqdm(os.listdir(args.output), desc="Clearing intermediate files"):
            subdir_path = os.path.join(args.output, subdir)
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if filename == "vocals.wav" or filename == "transcription.txt":
                    os.unlink(file_path)  # Delete the file
                if filename == "Clips_turns":
                    shutil.rmtree(file_path)
        print("    Intermediate files cleared.")



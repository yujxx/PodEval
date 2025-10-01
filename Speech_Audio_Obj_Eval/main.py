import os
import json
import glob
import csv
import shutil
import argparse
from tqdm import tqdm
from speech_metrics import (
    calculate_dnsmos, analyze_timbre_differences,
    calculate_wer, calculate_speaker_similarity
)
from audio_metrics import calculate_loudness, calculate_smr_for_speech_segments, calculate_casp_for_single_audio
from utils import split_audio_by_speaker, separate_audio_with_demucs, cluster_by_refspk
from models import (
    release_demucs_model, release_diarization_pipeline, 
    release_embedding_inference, release_casp_model, 
    release_onnx_sessions, release_all_models
)

def process_reference_file(reference_file, output_dir):
    if not os.path.exists(reference_file):
        print(f"Reference file not found: {reference_file}")
        return None, None, None
    
    reference_info = json.load(open(reference_file))
    ref_spk_dir = None
    gt_texts = None
    sent2refspk = None
    if "role_mapping" in reference_info:
        # get the reference audio files and rename it as segment_Y_SPEAKER_X.wav
        ref_spk_dir = f"{output_dir}/ref_spk_audio"
        os.makedirs(ref_spk_dir, exist_ok=True)
        for speaker, info in reference_info["role_mapping"].items():
            ref_audio = info["ref_audio"]
            ref_audio_name = f"segment_ref_{speaker}.wav"
            shutil.copy(ref_audio, os.path.join(ref_spk_dir, ref_audio_name))

    if "dialogue" in reference_info:
        dialogue = reference_info["dialogue"]
        gt_texts = ""
        sent2refspk = {}
        for sent_id, turn in enumerate(dialogue):
            if "text" in turn:
                gt_texts = f'{gt_texts}{turn["text"]} '
            else:
                print("Miss speaking content.")
            
            if "role" in turn: 
                formatted_sent_id = f"{sent_id:03d}"
                sent2refspk[formatted_sent_id] = turn["role"]

    return ref_spk_dir, gt_texts, sent2refspk

def process_audio_files_by_module(input_dir, output_dir, seperate, reference_dir="", segment_dir="", remove_intermediate=True):
    """
    Process all `.wav` files in the input directory by module, save their results to a CSV file.
    
    This function processes all files through each module before moving to the next module:
    1. Prepare file information, check existing results, and process reference files
    2. Audio separation (if needed) 
    3. Speaker segmentation
    4. Speaker similarity (if reference provided)
    5. Timbre differences
    6. DNSMOS calculation
    7. Loudness analysis
    8. WER calculation (if reference provided)
    9. CASP and SMR calculation (if music separation enabled)
    10. Clean up intermediate files
    11. Save individual results and create CSV
    
    Memory optimization: Each module releases its models from memory after completion to reduce memory usage.

    Args:
        input_dir (str): Path to the directory containing `.wav` files.
        output_dir (str): Path to the directory where individual file outputs will be saved.
        seperate (bool): Whether to separate the audio file with Demucs.
        reference_dir (str): Path to the directory containing reference information.
        segment_dir (str): Path to the directory containing segment files.
        remove_intermediate (bool): Whether to remove intermediate files after processing.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "all_results.csv")

    # Get all wav files
    wav_files = [f.replace(".wav", "") for f in os.listdir(input_dir) if f.endswith(".wav")]
    if not wav_files:
        print("No `.wav` files found in the input directory.")
        return

    print(f"Found {len(wav_files)} audio files to process.")
    
    # Initialize results storage
    all_results = {}
    file_info = {}  # Store file paths and metadata for each file
    
    # Step 1: Prepare file information, check existing results, and process reference files
    print("\n=== Step 1: Preparing file information ===")
    reference_data = {}  # Store processed reference data for each file
    
    # 新增：记录哪些文件已存在results.json
    files_with_results = set()

    for filename in tqdm(wav_files, desc="Preparing files"):
        filename = os.path.splitext(filename)[0]
        audio_file_path = os.path.join(input_dir, filename + ".wav")
        file_output_dir = os.path.join(output_dir, filename)
        reference_file = ""
        subsegment_dir = ""
        
        if reference_dir:
            reference_file = f"{reference_dir}/{filename}.json"
        if segment_dir:
            subsegment_dir = f"{segment_dir}/{filename}"

        # Check if results already exist
        output_file = os.path.join(file_output_dir, "results.json")
        if os.path.exists(output_file):
            print(f"Results already exist for {filename}, loading...")
            with open(output_file, "r", encoding="utf-8") as file:
                all_results[filename] = json.load(file)
            # 记录该文件已存在结果
            files_with_results.add(filename)
        else:
            all_results[filename] = {}
            
        file_info[filename] = {
            'audio_file_path': audio_file_path,
            'output_dir': file_output_dir,
            'reference_file': reference_file,
            'segment_dir': subsegment_dir
        }
        
        # Process reference file if it exists
        if reference_file and os.path.exists(reference_file):
            print(f"Processing reference file for {filename}...")
            ref_spk_dir, gt_texts, sent2refspk = process_reference_file(reference_file, file_output_dir)
            reference_data[filename] = {
                'ref_spk_dir': ref_spk_dir,
                'gt_texts': gt_texts,
                'sent2refspk': sent2refspk
            }
        else:
            reference_data[filename] = {
                'ref_spk_dir': None,
                'gt_texts': None,
                'sent2refspk': None
            }

    # Step 2: Audio separation (if needed)
    if seperate:
        print("\n=== Step 2: Audio separation with Demucs ===")
        for filename in tqdm(wav_files, desc="Separating audio"):
            if filename in files_with_results:
                continue  # 跳过已存在结果的文件
            if 'vocals_file' not in all_results[filename]:
                info = file_info[filename]
                print(f"Separating audio for {filename}...")
                vocals_file, accompaniment_file = separate_audio_with_demucs(
                    info['audio_file_path'], info['output_dir']
                )
                all_results[filename]['vocals_file'] = vocals_file
                all_results[filename]['accompaniment_file'] = accompaniment_file
            else:
                print(f"Audio separation already done for {filename}")
        
        # Release Demucs model after audio separation is complete
        release_demucs_model()

    # Step 3: Speaker segmentation
    print("\n=== Step 3: Speaker segmentation ===")
    for filename in tqdm(wav_files, desc="Segmenting speakers"):
        if filename in files_with_results:
            continue  # 跳过已存在结果的文件
        if 'segment_dir' not in all_results[filename]:
            info = file_info[filename]
            if info['segment_dir'] == "":
                print(f"Segmenting speakers for {filename}...")
                vocals_file = all_results[filename].get('vocals_file', info['audio_file_path'])
                segment_dir = f"{info['output_dir']}/vocal_segments"
                if not os.path.exists(segment_dir):
                    split_audio_by_speaker(vocals_file, segment_dir)
                if reference_data[filename]['ref_spk_dir']:
                    cluster_by_refspk(segment_dir, reference_data[filename]['ref_spk_dir'])
                all_results[filename]['segment_dir'] = segment_dir
            else:
                all_results[filename]['segment_dir'] = info['segment_dir']
        else:
            print(f"Speaker segmentation already done for {filename}")
    
    # Release diarization pipeline after speaker segmentation is complete
    release_diarization_pipeline()

    # Step 4: Speaker similarity (if reference provided)
    print("\n=== Step 4: Speaker similarity calculation ===")
    for filename in tqdm(wav_files, desc="Calculating speaker similarity"):
        if filename in files_with_results:
            continue  # 跳过已存在结果的文件
        if 'SIM' not in all_results[filename]:
            ref_data = reference_data[filename]
            if ref_data['ref_spk_dir'] and ref_data['sent2refspk']:
                print(f"Calculating speaker similarity for {filename}...")
                sim_result = calculate_speaker_similarity(
                    all_results[filename]['segment_dir'], ref_data['ref_spk_dir'], ref_data['sent2refspk']
                )
                all_results[filename]['SIM'] = sim_result
            else:
                print(f"No valid reference information for {filename}, skipping speaker similarity")

    # Step 5: Timbre differences
    print("\n=== Step 5: Timbre differences analysis ===")
    for filename in tqdm(wav_files, desc="Analyzing timbre differences"):
        if filename in files_with_results:
            continue  # 跳过已存在结果的文件
        if 'SPTD' not in all_results[filename]:
            print(f"Analyzing timbre differences for {filename}...")
            ref_data = reference_data[filename]
            if ref_data['ref_spk_dir']:
                spk_diff_info = analyze_timbre_differences(ref_data['ref_spk_dir'])
            else:
                spk_diff_info = analyze_timbre_differences(all_results[filename]['segment_dir'])
            all_results[filename]['SPTD'] = spk_diff_info
    
    # Release embedding inference after timbre analysis is complete
    release_embedding_inference()

    # Step 6: DNSMOS calculation
    print("\n=== Step 6: DNSMOS calculation ===")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "DNSMOS", "DNS-Challenge", "DNSMOS")
    
    for filename in tqdm(wav_files, desc="Calculating DNSMOS"):
        if filename in files_with_results:
            continue  # 跳过已存在结果的文件
        if 'DNSMOS(OVRL)' not in all_results[filename]:
            print(f"Calculating DNSMOS for {filename}...")
            dnsmos_scores = calculate_dnsmos(all_results[filename]['segment_dir'], model_path)
            if dnsmos_scores:
                all_results[filename]["DNSMOS(OVRL)"] = dnsmos_scores['OVRL']
                all_results[filename]["DNSMOS(SIG)"] = dnsmos_scores['SIG']
                all_results[filename]["DNSMOS(BAK)"] = dnsmos_scores['BAK']
                all_results[filename]["DNSMOS(P808_MOS)"] = dnsmos_scores['P808_MOS']
            else:
                print(f"DNSMOS calculation failed for {filename}")
    
    # Release ONNX sessions after DNSMOS calculation is complete
    release_onnx_sessions()

    # Step 7: Loudness analysis
    print("\n=== Step 7: Loudness analysis ===")
    for filename in tqdm(wav_files, desc="Analyzing loudness"):
        if filename in files_with_results:
            continue  # 跳过已存在结果的文件
        if 'LOUD_IT' not in all_results[filename]:
            print(f"Analyzing loudness for {filename}...")
            info = file_info[filename]
            loudness_result = calculate_loudness(info['audio_file_path'])
            if loudness_result:
                all_results[filename]["LOUD_IT"] = loudness_result['Integrated Loudness (LUFS)']
                all_results[filename]["LOUD_TP"] = loudness_result['True Peak (dBTP)']
                all_results[filename]["LOUD_LRA"] = loudness_result['Loudness Range (LRA, LU)']
                all_results[filename]["LOUD_IT_SCORE"] = loudness_result['Scores']['Loudness Score']
                all_results[filename]["LOUD_TP_SCORE"] = loudness_result['Scores']['True Peak Score']
                all_results[filename]["LOUD_LRA_SCORE"] = loudness_result['Scores']['LRA Score']
                all_results[filename]["LOUD_AVG_SCORE"] = loudness_result['Scores']['Overall Score']
            else:
                print(f"Loudness analysis failed for {filename}")

    # Step 8: WER calculation (if reference provided)
    print("\n=== Step 8: WER calculation ===")
    for filename in tqdm(wav_files, desc="Calculating WER"):
        if filename in files_with_results:
            continue  # 跳过已存在结果的文件
        if 'WER' not in all_results[filename]:
            ref_data = reference_data[filename]
            if ref_data['gt_texts']:
                print(f"Calculating WER for {filename}...")
                info = file_info[filename]
                vocals_file = all_results[filename].get('vocals_file', info['audio_file_path'])
                wer_result = calculate_wer(vocals_file, ref_data['gt_texts'], info['output_dir'])
                if wer_result:
                    all_results[filename]["WER"] = wer_result['WER']
                else:
                    print(f"WER calculation failed for {filename}")
            else:
                print(f"No ground truth text for {filename}, skipping WER")

    # Step 9: CASP and SMR calculation (if music separation enabled)
    if seperate:
        print("\n=== Step 9: CASP and SMR calculation ===")
        casp_model_path = os.path.join(script_dir, "casp")
        
        for filename in tqdm(wav_files, desc="Calculating CASP and SMR"):
            if filename in files_with_results:
                continue  # 跳过已存在结果的文件
            if 'CASP' not in all_results[filename]:
                print(f"Calculating CASP and SMR for {filename}...")
                vocals_file = all_results[filename].get('vocals_file')
                accompaniment_file = all_results[filename].get('accompaniment_file')
                
                if vocals_file and accompaniment_file:
                    # CASP calculation
                    casp_result = calculate_casp_for_single_audio(vocals_file, accompaniment_file, model_path=casp_model_path)
                    if casp_result:
                        all_results[filename]["CASP"] = casp_result['casp_score']
                    
                    # SMR calculation
                    smr_result = calculate_smr_for_speech_segments(vocals_file, accompaniment_file)
                    if smr_result:
                        all_results[filename]["SMR_MIN"] = smr_result['smr_min']
                        all_results[filename]["SMR_MAX"] = smr_result['smr_max']
                        all_results[filename]["SMR_AVG"] = smr_result['smr_avg']
                        all_results[filename]["SMR_BASIC_SCORE"] = smr_result['smr_basic_score']
                        all_results[filename]["SMR_REF_SCORE_0.1_0.01"] = smr_result['smr_ref_score']
                else:
                    print(f"Missing vocals or accompaniment file for {filename}")
        
        # Release CASP model after CASP and SMR calculation is complete
        release_casp_model()

    # Step 10: Clean up intermediate files
    if remove_intermediate:
        print("\n=== Step 10: Cleaning up intermediate files ===")
        for filename in tqdm(wav_files, desc="Cleaning up"):
            info = file_info[filename]
            
            # Remove separated audio files
            if seperate:
                vocals_file = all_results[filename].get('vocals_file')
                accompaniment_file = all_results[filename].get('accompaniment_file')
                if vocals_file and os.path.exists(vocals_file):
                    os.remove(vocals_file)
                if accompaniment_file and os.path.exists(accompaniment_file):
                    os.remove(accompaniment_file)

            # Remove segment files
            segment_dir = all_results[filename].get('segment_dir')
            if segment_dir and segment_dir == f"{info['output_dir']}/vocal_segments":
                for file in glob.glob(os.path.join(segment_dir, "*.wav")):
                    os.remove(file)
                if os.path.exists(segment_dir):
                    os.rmdir(segment_dir)

    # Step 11: Save individual results and create CSV
    print("\n=== Step 11: Saving results ===")
    data_rows = []
    
    for filename in tqdm(wav_files, desc="Saving results"):
        # Save individual result file
        info = file_info[filename]
        output_file = os.path.join(info['output_dir'], "results.json")
        
        # Remove internal file paths from results before saving
        result_to_save = {k: v for k, v in all_results[filename].items() 
                         if not k.endswith('_file') and k != 'segment_dir'}
        
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(result_to_save, file, indent=4, ensure_ascii=False)
        
        # Prepare row for CSV
        row = {"filename": filename}
        row.update(result_to_save)
        data_rows.append(row)

    # Write CSV file
    if data_rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["filename"] + list(data_rows[0].keys())[1:]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_rows)
        print(f"Processing completed. Results saved to {output_csv}")
    else:
        print("No results to write. No `.wav` files were processed.")
    
    # Final memory cleanup
    print("\n=== Final memory cleanup ===")
    release_all_models()
    print("All processing completed and memory cleaned up.")

# Keep the original function for backward compatibility
def process_single_audio(audio_file, output_dir, seperate, reference_file="", segment_dir="", remove_intermediate=False):
    """
    Process a single audio file.
    """
    # check if the output file exists, avoid re-calculating the same file
    output_file = os.path.join(output_dir, "results.json")
    if os.path.exists(output_file):
        print(f"Results already exist at: {output_file}")
        with open(output_file, "r", encoding="utf-8") as file:
            return json.load(file)

    result = {}
    
    # if with background music, separate it with Demucs
    vocals_file = audio_file
    accompaniment_file = None
    if seperate:    
        print("*With music* Separate audio using Demucs...")
        vocals_file, accompaniment_file = separate_audio_with_demucs(audio_file, output_dir)
    else:
        print("*Without music* Using the original audio file as vocals.")
        
    # if reference file is provided, load the reference information
    ref_spk_dir, gt_texts, sent2refspk = process_reference_file(reference_file, output_dir)

    if segment_dir == "":
        print("*No segment files provided* Splitting vocal file into multiple segments based on speakers...")
        segment_dir = f"{output_dir}/vocal_segments"
        if not os.path.exists(segment_dir):
            split_audio_by_speaker(vocals_file, segment_dir)
    else:
        print(f"[SIM]: Calculating speaker similarity score...")
        if ref_spk_dir and sent2refspk:
            sim_result = calculate_speaker_similarity(segment_dir, ref_spk_dir, sent2refspk)
            result["SIM"] = sim_result
        else:
            print("\tNo reference file provided, skip speaker similarity score.")

    print("\n[SPTD]: Evaluating timbre differences...")
    if ref_spk_dir:
        spk_diff_info = analyze_timbre_differences(ref_spk_dir)
    else:
        spk_diff_info = analyze_timbre_differences(segment_dir)
    result["SPTD"] = spk_diff_info

    print("\n[DNSMOS]: Calculate the DNSMOS...")
    # Use absolute path to ensure the model files can be found
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "DNSMOS", "DNS-Challenge", "DNSMOS")
    dnsmos_scores = calculate_dnsmos(segment_dir, model_path)
    if dnsmos_scores:
        result["DNSMOS(OVRL)"] = dnsmos_scores['OVRL']
        result["DNSMOS(SIG)"] = dnsmos_scores['SIG']
        result["DNSMOS(BAK)"] = dnsmos_scores['BAK']
        result["DNSMOS(P808_MOS)"] = dnsmos_scores['P808_MOS']
    
    print("\n[LDA]: Loudness Analysis...")
    loudness_result = calculate_loudness(audio_file)
    if loudness_result:
        result["LOUD_IT"] = loudness_result['Integrated Loudness (LUFS)']
        result["LOUD_TP"] = loudness_result['True Peak (dBTP)']
        result["LOUD_LRA"] = loudness_result['Loudness Range (LRA, LU)']
        result["LOUD_IT_SCORE"] = loudness_result['Scores']['Loudness Score']
        result["LOUD_TP_SCORE"] = loudness_result['Scores']['True Peak Score']
        result["LOUD_LRA_SCORE"] = loudness_result['Scores']['LRA Score']
        result["LOUD_AVG_SCORE"] = loudness_result['Scores']['Overall Score']

    if gt_texts:
        print("[WER]: Calculate WER...")
        wer_result = calculate_wer(vocals_file, gt_texts, output_dir)
        if wer_result:
            result["WER"] = wer_result['WER']

    if seperate:    
        print("[CASP]: Calculate Speech-to-Music CASP score...")
        model_path = os.path.join(script_dir,"casp")
        casp_result = calculate_casp_for_single_audio(vocals_file, accompaniment_file, model_path=model_path)
        if casp_result:
            result["CASP"] = casp_result['casp_score']

        print("[SMR]: Calculate Speech-to-Music Ratio SMR...")
        smr_result = calculate_smr_for_speech_segments(vocals_file, accompaniment_file)
        if smr_result:
            result["SMR_MIN"] = smr_result['smr_min']
            result["SMR_MAX"] = smr_result['smr_max']
            result["SMR_AVG"] = smr_result['smr_avg']
            result["SMR_BASIC_SCORE"] = smr_result['smr_basic_score']
            result["SMR_REF_SCORE_0.1_0.01"] = smr_result['smr_ref_score']
    
    # Remove intermediate files if specified
    if remove_intermediate:
        if seperate:
            if vocals_file and os.path.exists(vocals_file):
                os.remove(vocals_file)
            if accompaniment_file and os.path.exists(accompaniment_file):
                os.remove(accompaniment_file)

        if segment_dir == f"{output_dir}/vocal_segments":    
            for file in glob.glob(os.path.join(segment_dir, "*.wav")):
                os.remove(file)
            os.rmdir(segment_dir)

    # Print results
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    # 返回所有结果
    return result

def process_audio_files_to_csv(input_dir, output_dir, seperate, reference_dir="", segment_dir="", remove_intermediate=True):
    """
    Process all `.wav` files in the input directory, save their results to a CSV file.

    Args:
        input_dir (str): Path to the directory containing `.wav` files.
        output_dir (str): Path to the directory where individual file outputs will be saved.
        seperate (bool): Whether to separate the audio file with Demucs.
        reference_dir (str): Path to the directory containing reference information.
        segment_dir (str): Path to the directory containing segment files.
        remove_intermediate (bool): Whether to remove intermediate files after processing.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "all_results.csv")

    # Initialize a list to store rows of data for the CSV
    data_rows = []

    # Iterate through all files in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".wav"):  # Process only `.wav` files
            filename = os.path.splitext(filename)[0]
            audio_file_path = os.path.join(input_dir, filename + ".wav")
            file_output_dir = os.path.join(output_dir, filename)
            reference_file = ""
            subsegment_dir = ""
            if reference_dir:
                reference_file = f"{reference_dir}/{filename}.json"
            if segment_dir:
                subsegment_dir = f"{segment_dir}/{filename}"

            results = process_single_audio(audio_file_path, file_output_dir, seperate, reference_file, subsegment_dir, remove_intermediate)

            row = {"filename": filename}
            row.update(results)
            data_rows.append(row)

    # Write all results to the CSV file
    if data_rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            # Get the header from the keys of the first result
            fieldnames = ["filename"] + list(data_rows[0].keys())[1:]  # Ensure "filename" is the first column
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header and data rows
            writer.writeheader()
            writer.writerows(data_rows)

        print(f"Processing completed. Results saved to {output_csv}")
    else:
        print("No results to write. No `.wav` files were processed.")
    

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audio Processing Script")
    parser.add_argument("--input", required=True, help="Path to input audio directory")
    parser.add_argument("--output", required=True, help="Directory to save results")
    parser.add_argument("--with_music", action='store_true', help="If podcasts contain background music, set this arg to separate it with Demucs.")
    parser.add_argument("--mode", default="file", help="file: process file by file; module: process all files through each module before moving to the next module")
    parser.add_argument("--ref_info", default="", help="Reference information, directory of json files containing reference texts / speaker information.")
    parser.add_argument("--segment_dir", default="", help="Directory of provided segment files.")
    parser.add_argument("--remove_itm", action='store_true', help="Remove intermediate files after processing")

    args = parser.parse_args()

    import time
    start_time = time.time()
    if args.mode == "file":
        process_audio_files_to_csv(args.input, args.output, args.with_music, args.ref_info, args.segment_dir, remove_intermediate=args.remove_itm)
    elif args.mode == "module":
        process_audio_files_by_module(args.input, args.output, args.with_music, args.ref_info, args.segment_dir, remove_intermediate=args.remove_itm)
    else:
        print("Invalid mode, please choose from 'file' or 'module'")
    end_time = time.time()
    print(f"Mode: {args.mode} - Time taken: {end_time - start_time} seconds")

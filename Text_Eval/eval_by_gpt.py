import os
import json5
import argparse
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm

os.environ['EVAL_OPENAI_KEY']='xxx'
os.environ["OPENAI_BASE_URL"] = 'xxx'

import glob
import pickle
import time
os.environ["USE_OPENAI_CACHE"]='True'
USE_OPENAI_CACHE = os.environ.get('USE_OPENAI_CACHE', False)
openai_cache = []
if USE_OPENAI_CACHE:
    os.makedirs('cache', exist_ok=True)
    for cache_file in glob.glob('cache/*.pkl'):
        with open(cache_file, 'rb') as file:
            openai_cache.append(pickle.load(file))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='./eval_dialogue_compare.prompt', help='prompt file path')
    parser.add_argument('--input-dir-1', type=str, default='', help='Directory 1 with files to be evaluated')
    parser.add_argument('--input-dir-2', type=str, default='', help='Directory 2 with files to be evaluated')
    parser.add_argument('--output-dir', type=str, default='', help='Where to save the output result file.')
    return parser.parse_args()

def extract_substring_with_quotes(input_string, quotes="'''"):
    pattern = f"{quotes}(.*?){quotes}"
    matches = re.findall(pattern, input_string, re.DOTALL)
    for i in range(len(matches)):
        if matches[i][:4] == 'json':
            matches[i] = matches[i][4:]
    
    if len(matches) == 1:
        return matches[0]
    else:
        return matches


def try_extract_content_from_quotes(content):
    if "'''" in content:
        return extract_substring_with_quotes(content)
    elif "```" in content:
        return extract_substring_with_quotes(content, quotes="```")
    else:
        return content
    
def chat_with_gpt(prompt, api_key, sys_info="You are a helpful assistant."):
    if USE_OPENAI_CACHE:
        filtered_object = list(filter(lambda x: x['prompt'] == (sys_info + prompt), openai_cache))
        if len(filtered_object) > 0:
            response = filtered_object[0]['response']
            return response
        
    client = OpenAI(api_key=api_key, timeout=240.0, max_retries=0)
    chat = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": sys_info
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    if USE_OPENAI_CACHE:
        cache_obj = {
            'prompt': sys_info + prompt,
            'response': chat.choices[0].message.content
        }
        cache_file = f'cache/{time.time()}.pkl'
        with open(cache_file, 'wb') as _openai_cache:
            pickle.dump(cache_obj, _openai_cache)
            openai_cache.append(cache_obj)

    return chat.choices[0].message.content

def calculate_statistics(df):

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        return {
            'mean': df[numeric_columns].mean().to_dict(),
            'std': df[numeric_columns].std().to_dict(),
            'min': df[numeric_columns].min().to_dict(),
            'max': df[numeric_columns].max().to_dict(),
            'median': df[numeric_columns].median().to_dict()
        }
def print_line(file, line):
    print(line)
    file.write(line + "\n")


def generate_report(dir1, dir2, results, results_reverse, file_path):
    with open(file_path, 'w') as file:
        # Header information
        print_line(file, f"Dir1: {dir1}")
        print_line(file, f"Dir2: {dir2}\n")
        print_line(file, "--- Statistics Report ---\n")

        # Helper function to output a table for a given statistics dictionary
        def output_statistics_table(title, statistics):
            print_line(file, f"{title}\n")
            print_line(file, f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
            print_line(file, "-" * 70)
            metrics = statistics["mean"].keys()
            for metric in metrics:
                mean = statistics["mean"].get(metric, 0)
                std = statistics.get("std", {}).get(metric, 0)
                min_val = statistics.get("min", {}).get(metric, 0)
                max_val = statistics.get("max", {}).get(metric, 0)
                median_val = statistics.get("median", {}).get(metric, 0)
                print_line(file, f"{metric:<20} {mean:<10.4f} {std:<10.4f} {min_val:<10.4f} {max_val:<10.4f} {median_val:<10.4f}")
            print_line(file, "\n")

        # Output overall statistics for results
        output_statistics_table("*** OVERALL (Dir1 vs Dir2) ***", results['overall_statistics'])

        # Output overall statistics for results_reverse
        output_statistics_table("*** OVERALL (Reverse) ***", results_reverse['overall_statistics'])

        # Output Mean comparison table for overall statistics
        print_line(file, "*** Mean Comparison Table (Overall) ***\n")
        header = f"{'Metric':<20} | {'Dir1 vs Dir2':<10} | {'Reverse':<10} | {'Mean':<10}"
        print_line(file, header)
        print_line(file, "-" * len(header))
        for metric, mean_value in results['overall_statistics']["mean"].items():
            reverse_value = results_reverse['overall_statistics']["mean"].get(metric, 0)
            avg_value = (mean_value - reverse_value) / 2
            print_line(file, f"{metric:<20} | {mean_value:<10.4f} | {reverse_value:<10.4f} | {avg_value:<10.4f}")
        print_line(file, "\n")

        # Output category statistics
        for cat in results['category_statistics']:
            print_line(file, f"*** {cat.capitalize()} Statistics ***\n")
            # Output category statistics for results
            output_statistics_table(f"--- {cat.capitalize()} (Dir1 vs Dir2) ---", results['category_statistics'][cat])

            # Output category statistics for results_reverse
            output_statistics_table(f"--- {cat.capitalize()} (Reverse) ---", results_reverse['category_statistics'][cat])

            # Output Mean comparison table for category statistics
            print_line(file, f"*** Mean Comparison Table ({cat.capitalize()}) ***\n")
            header = f"{'Metric':<20} | {'Dir1 vs Dir2':<10} | {'Reverse':<10} | {'Mean':<10}"
            print_line(file, header)
            print_line(file, "-" * len(header))
            for metric, mean_value in results['category_statistics'][cat]["mean"].items():
                reverse_value = results_reverse['category_statistics'][cat]["mean"].get(metric, 0)
                avg_value = (mean_value - reverse_value) / 2
                print_line(file, f"{metric:<20} | {mean_value:<10.4f} | {reverse_value:<10.4f} | {avg_value:<10.4f}")
            print_line(file, "\n")

        # Output file2reason details
        print_line(file, "\n--- File-Level Comparisons ---\n")
        for key in results['file2reason']:
            print_line(file, f"**File: {key}**")
            reason_text = "\n".join([f"{k}: {v}" for k, v in results['file2reason'][key].items()])
            print_line(file, reason_text)
            print_line(file, "*Reverse*")
            reverse_reason_text = "\n".join([f"{k}: {v}" for k, v in results_reverse['file2reason'][key].items()])
            print_line(file, reverse_reason_text)
            print_line(file, "")  # Add an empty line for spacing
            
class TextEvaluator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def load_prompt(self, prompt_path: str) -> str:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def load_text(self, text_path: str) -> str:
        with open(text_path, 'r', encoding='utf-8') as f:
            texts = ''
            for line in f:
                json_data = json5.loads(line.strip())
                texts = f'{texts}{json_data["speaker"]}: {json_data["speaking_content"]}\n'
        return texts
            
    def parse_gpt_response(self, response: str) -> dict:
        response_json = json5.loads(response)
        
        return response_json

    def compare_two_texts(self, prompt: str, text1: str, text2: str) -> dict:
        sys_info = "You are a helpful assistant that evaluates text quality."

        prompt = prompt.replace('${dialogue1_to_be_evaluated}', text1)
        complete_prompt = prompt.replace('${dialogue2_to_be_evaluated}', text2)
        
        response = try_extract_content_from_quotes(chat_with_gpt(complete_prompt, self.api_key, sys_info))
        
        return self.parse_gpt_response(response)

    
    def evaluate_dirs_compare(self, prompt_path: str, dir_path_1: str, dir_path_2: str):
        prompt = self.load_prompt(prompt_path)
        
        results = []
        file2reason = {}
        cat2results = {}
        file_list = [f for f in os.listdir(dir_path_1) if os.path.exists(os.path.join(dir_path_2, f))]
        for file in tqdm(file_list):
            file1_path = os.path.join(dir_path_1, file)
            file2_path = os.path.join(dir_path_2, file)

            text1 = self.load_text(file1_path)
            text2 = self.load_text(file2_path)
            result = self.compare_two_texts(prompt, text1, text2)
            
            results.append(result)
            file2reason[file] = result
            
            category = file.split('_')[0]
            if category not in cat2results:
                cat2results[category]=[result]
            else:
                cat2results[category].append(result)

        df_all = pd.DataFrame(results)
        statics = calculate_statistics(df_all)
        
        # Calculate statistics for each category
        cat2statistics = {}
        for category, category_results in cat2results.items():
            df_category = pd.DataFrame(category_results)
            cat2statistics[category] = calculate_statistics(df_category)
        
        return {
            "overall_statistics": statics,
            "category_statistics": cat2statistics,
            "file2reason": file2reason,
        }
    
    
def main():
    args = parse_arguments()
    prompt_file = args.prompt
    output_file = f'{args.output_dir}/gpt_eval.csv'

    api_key = os.environ.get('EVAL_OPENAI_KEY')      
    evaluator = TextEvaluator(api_key)

    print(f'Dir 1: {args.input_dir_1}')
    print(f'Dir 2: {args.input_dir_2}')

    results = evaluator.evaluate_dirs_compare(
        prompt_path=prompt_file,
        dir_path_1=args.input_dir_1,
        dir_path_2=args.input_dir_2,
    )
    #reverse evaluation
    results_reverse = evaluator.evaluate_dirs_compare(
        prompt_path=prompt_file,
        dir_path_1=args.input_dir_2,
        dir_path_2=args.input_dir_1,
    )

    generate_report(args.input_dir_1, args.input_dir_2, results, results_reverse, output_file)

    

if __name__ == "__main__":
    main()
    
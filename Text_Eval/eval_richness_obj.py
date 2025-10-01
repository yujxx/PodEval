import os
import argparse
import json5
import numpy as np
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, sent_tokenize
import jieba
import jieba.posseg as pseg
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import logging
import re
import nltk
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='english', help='The text languange: english or chinese.')
    parser.add_argument('--input_dir', type=str, default='', help='The directory path of conversation files to be evaluated.')
    parser.add_argument('--output_dir', type=str, default='', help='Where to save the output result file.')
    return parser.parse_args()

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json5.load(file)
            texts = ''
            for turn in json_data:
                if "speaking_content" in turn:
                    texts = f'{texts}{turn["speaking_content"]} '
                elif "text" in turn:
                    texts = f'{texts}{turn["text"]} '
                else:
                    print("Miss speaking content.")
            return texts
        
    except Exception as e:
        print(f"Read line by line.")
    
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = ''
            for line in f:
                json_data = json5.loads(line.strip())
                texts = f'{texts}{json_data["speaking_content"]} '
        return texts

class ContentRichnessEvaluator:
    def __init__(self, lang='english'):
        self.lang = lang

        self.stopwords = {
            'chinese': self._load_chinese_stopwords(),
            'english': set(stopwords.words('english'))
        }
        
        self.bert_tokenizers = {
            'chinese': AutoTokenizer.from_pretrained('bert-base-chinese'),
            'english': AutoTokenizer.from_pretrained('bert-base-uncased')
        }
        self.bert_models = {
            'chinese': AutoModel.from_pretrained('bert-base-chinese'),
            'english': AutoModel.from_pretrained('bert-base-uncased')
        }
        
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    def _load_chinese_stopwords(self):
        return set(['的', '了', '和', '是', '就', '都', '而', '及', '与', '这'])
    
    def tokenize(self, text):
        if self.lang == 'chinese':
            return list(jieba.cut(text))
        return word_tokenize(text.lower())
    
    def sentence_split(self, text):
        if self.lang == 'chinese':
            return re.split('[。！？]', text)
        return sent_tokenize(text)
    
    def get_pos_tags(self, tokens):
        if self.lang == 'chinese':
            words = pseg.cut(''.join(tokens))
            return [(word, tag) for word, tag in words]
        return pos_tag(tokens)

    def calculate_semantic_diversity(self, text, window_size=100):
        tokens = self.tokenize(text)
        if len(tokens) < window_size * 2:
            return 0.0
        
        windows = [' '.join(tokens[i:i+window_size]) 
                  for i in range(0, len(tokens)-window_size, window_size)]
        if len(windows) < 2:
            return 0.0
        
        tokenizer = self.bert_tokenizers[self.lang]
        model = self.bert_models[self.lang]
        
        embeddings = []
        for window in windows:
            inputs = tokenizer(window, return_tensors='pt', 
                             padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
        
        distances = []
        embeddings = np.array(embeddings)
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                cos_sim = np.dot(embeddings[i], embeddings[j]) / \
                         (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                distances.append(1 - cos_sim)
        
        return np.mean(distances)
    
    
    def calculate_distinct_n_with_sliding_window(self, text, n, window_size=100):
        tokens = self.tokenize(text)
        if len(tokens) < n:
            return 0.0

        distinct_n_values = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            n_grams = list(ngrams(window, n))
            
            if not n_grams:
                continue
                
            distinct_n_grams = set(n_grams)
            distinct_n_value = len(distinct_n_grams) / len(n_grams)
            distinct_n_values.append(distinct_n_value)

        return sum(distinct_n_values) / len(distinct_n_values) if distinct_n_values else 0.0

    def calculate_vocabulary_richness(self, text):
        """Calculate vocabulary richness metrics (TTR and MATTR)"""
        tokens = self.tokenize(text)
        if not tokens:
            return {'TTR': 0.0, 'MATTR': 0.0}
            
        types = set(tokens)
        ttr = len(types) / len(tokens)
        
        # Calculate Moving-Average TTR (MATTR)
        window_size = min(100, len(tokens))
        if len(tokens) < window_size:
            return {'TTR': ttr, 'MATTR': ttr}
        
        mattrs = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i+window_size]
            window_ttr = len(set(window)) / window_size
            mattrs.append(window_ttr)
        
        return np.mean(mattrs) if mattrs else ttr

    def calculate_information_density(self, text):
        """Calculate information density using entropy"""
            
        tokens = self.tokenize(text)

        #filtered_tokens = tokens  
        filtered_tokens = [token for token in tokens if token not in self.stopwords[self.lang]]

        if not filtered_tokens:
            return 0.0
            
        freq = Counter(filtered_tokens)
        total = len(filtered_tokens)
        
        probs = [count/total for count in freq.values()]
        if not probs:
            return 0.0
            
        return -sum(p * np.log2(p) for p in probs)
    
    
    def evaluate_text(self, text):
        """Evaluate Single File"""

        basic_metrics = {
            'distinct_2': self.calculate_distinct_n_with_sliding_window(text, 2, 100),
            'information_density': self.calculate_information_density(text),
            'semantic_diversity': self.calculate_semantic_diversity(text),
            'MATTR': self.calculate_vocabulary_richness(text)
        }
    
        return basic_metrics

    def evaluate_directory(self, dir_path):
        all_results = []
        category_files = {}

        for file in tqdm(os.listdir(dir_path)):
            if file.endswith(".json"):
                category = file.split("_")[0]
                if category not in category_files:
                    category_files[category] = []

                file_path = os.path.join(dir_path, file)
                texts = read_json_file(file_path)
                results = self.evaluate_text(texts)
                
                results['file_name'] = file
                results['category'] = category
                
                all_results.append(results)
                category_files[category].append(results)
        
        # Overall statistics
        df_all = pd.DataFrame(all_results)
        overall_stats = self._calculate_statistics(df_all)

        # Category statistics
        category_stats = {}
        for category, results in category_files.items():
            df_category = pd.DataFrame(results)
            category_stats[category] = self._calculate_statistics(df_category)
        
        return {
            'overall_stats': overall_stats,
            'category_stats': category_stats,
            'detailed_results': df_all
        }
    
    def _calculate_statistics(self, df):
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

def print_and_save_report(stats, file_path='overall_statistics_report.txt'):
    with open(file_path, 'w') as file:
        # Print overall statistics
        report_title = "=== Overall Statistics Report ==="
        print_line(file, report_title)

        overall_stats = stats['overall_stats']
        # Collect the statistics into a formatted table
        header = "--- Mean ------ Std ------ Max ------ Median ---"
        print_line(file, header)

        # Extract relevant statistics and format them row by row
        for feature in overall_stats['mean'].keys():
            mean = overall_stats['mean'][feature]
            std = overall_stats['std'][feature]
            max_val = overall_stats['max'][feature]
            median = overall_stats['median'][feature]
            formatted_line = f"{feature}: {mean:.4f}    |    {std:.4f}    |    {max_val:.4f}    |    {median:.4f}"
            print_line(file, formatted_line)

        # Print category statistics
        category_stats = stats['category_stats']
        category_title = "\n=== Category Statistics ==="
        print_line(file, category_title)

        for category, cat_stats in category_stats.items():
            category_section_title = f"\n--- Category: {category} ---"
            print_line(file, category_section_title)
            print_line(file, header)
            for feature in cat_stats['mean'].keys():
                mean = cat_stats['mean'][feature]
                std = cat_stats['std'][feature]
                max_val = cat_stats['max'][feature]
                median = cat_stats['median'][feature]
                formatted_line = f"{feature}: {mean:.4f}    |    {std:.4f}    |    {max_val:.4f}    |    {median:.4f}"
                print_line(file, formatted_line)

        # Optionally print detailed results if needed
        detailed_results = stats.get('detailed_results')
        if detailed_results is not None:
            detailed_title = "\n=== Detailed Results ==="
            print_line(file, detailed_title)
            print_line(file, detailed_results.to_string())

    print(f"Report saved to {file_path}")



def main():

    args = parse_arguments()
    evaluator = ContentRichnessEvaluator()

    comparison_result = evaluator.evaluate_directory(args.input_dir)
    output_file = f'{args.output_dir}/obj_richness.csv'
    print_and_save_report(comparison_result, output_file)

if __name__ == "__main__":
    main()
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent_i = SentimentIntensityAnalyzer()
import pandas as pd
import numpy as np

import pandas as pd
import re

# IMPORT TAXONOMY OF first
def csv_to_dictionary(filename, encodings=['utf-8', 'latin-1', 'utf-16', 'windows-1252']):
    for encoding in encodings:
        try:
            csv_dict = pd.read_csv(filename, encoding=encoding)
            break
        except:
            continue
    csv_dict = csv_dict.T
    csv_dict.fillna('@remove@', inplace=True)
    csv_dict.index = csv_dict.index.str.lower()
    csv_dict = csv_dict.apply(lambda x: x.astype(str).str.lower())
    final_dict = dict(zip(csv_dict.index,csv_dict.values.tolist()))
    values = ['@remove@'] 
    final_dict ={i: [a for a in j if a not in values] for i,j in final_dict.items()}
    return final_dict

change_dict = csv_to_dictionary('Taxonomy for direction of change.csv')
change_dict = {key: [value for value in change_dict[key] if re.search(r'[a-zA-Z0-9]+', value)] for key in change_dict.keys()}

# IMPORT TAXONOMY OF CHANGE
def csv_to_dictionary_semicolon_delimiter(filename, encodings=['utf-8', 'latin-1', 'utf-16', 'windows-1252']):
    for encoding in encodings:
        try:
            csv_dict = pd.read_csv(filename, delimiter=';', encoding=encoding)
            break
        except:
            continue
    csv_dict = csv_dict.T
    csv_dict.fillna('@remove@', inplace=True)
    csv_dict.index = csv_dict.index.str.lower()
    csv_dict = csv_dict.apply(lambda x: x.astype(str).str.lower())
    final_dict = dict(zip(csv_dict.index,csv_dict.values.tolist()))
    values = ['@remove@'] 
    final_dict ={i: [a for a in j if a not in values] for i,j in final_dict.items()}
    return final_dict

all_topics = csv_to_dictionary_semicolon_delimiter('all_topics.csv')
all_topics = {key: [value for value in all_topics[key] if re.search(r'[a-zA-Z0-9]+', value)] for key in all_topics.keys()}
all_topics = {k: [x.strip() for x in v] for k, v in all_topics.items()}



# CLEAN TEXT INSIDE TEXT COLUMN
def clean_text(df): # Lowercase, clean and strip text.
    df["text"] = df.text.str.lower()
    df["text"] = df.text.str.replace("\ufeff", "")
    df["text"] = df.text.str.strip()

# SPLIT TEXT INTO SENTENCES AND EXPAND THE DATAFRAME
def split_sentences(df): 
    df["sentences"] = df["text"].apply(nltk.sent_tokenize)
    return df.explode("sentences")

# CHECK CO-OCCURRENCES FOR EACH COMBINATION OF TOPICS AND CHANGE & RETURN MINIMUM DISTANCE
def check_co_occurrences(row, first_dict, second_dict): 
    for first_key, first_words in first_dict.items():
        first_words = [re.escape(word) for word in first_words]
        first_pattern = r"\b(" + "|".join(first_words) + r")\b"
        for second_key, second_words in second_dict.items():
            second_words = [re.escape(word) for word in second_words]
            second_pattern = r"\b(" + "|".join(second_words) + r")\b"
            co_occurrences = []
            first_occurrences = [m.start() for m in re.finditer(first_pattern, row['sentences'])]
            second_occurrences = [m.start() for m in re.finditer(second_pattern, row['sentences'])]
            for first_occurrence in first_occurrences:
                for second_occurrence in second_occurrences:
                    start = min(first_occurrence, second_occurrence)
                    end = max(first_occurrence, second_occurrence)
                    num_words = len(row['sentences'][start:end].split())
                    co_occurrences.append(num_words)
            if co_occurrences:
                row[ first_key + ' [' + second_key + ']'] = min(co_occurrences)
            else:
                row[ first_key + ' [' + second_key + ']'] = np.nan
    return row

# FILTER RESULTS BY SELECTED DISTANCE
def update_columns(df, distance):
    for column in df.columns[7:]: #change 7 if you add columns or vadar_compound
        df[column] = df[column].apply(lambda x: 1 if x <= distance else np.nan) # if x <= distance change value to 1 else NaN
    return df

# GROUP SENTENCES BY DOCUMENT
def group_and_sum(df):
    result = (
        df.groupby(["file_name", "nid", "country", "title", "from_date"])
        .sum()
        .reset_index()
    )
    return result

def text_analysis(folder_path, distance, all_topics, change_dict):
    # Get a list of all tsv files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]
    for file in files:
        # Read the file
        df = pd.read_csv(os.path.join(folder_path, file), sep='\t')
        # Perform the operations on the dataframe
        df = df[['nid','country', 'title','file_name','from_date','national overview']].copy().dropna()
        df = df.rename(columns={'national overview': 'text'})
        clean_text(df)
        df = split_sentences(df)
        df.drop('text', axis=1, inplace=True)
        df = df.apply(check_co_occurrences, axis=1, args=(all_topics, change_dict))
        df = update_columns(df, distance)
        codebook = group_and_sum(df)
        # Save the output to a new file, using the original file name
        base_name = os.path.splitext(file)[0]
        output_folder = 'countries_output'
        os.makedirs(output_folder, exist_ok=True)
        #df.to_csv(os.path.join(output_folder, f'{base_name}_sentences.tsv'), sep='\t', index=False) # optional dataframe that saves all sentences
        codebook.to_csv(os.path.join(output_folder, f'{base_name}_codebook.tsv'), sep='\t', index=False)


def load_taxonomy():
    change_dict = csv_to_dictionary('Taxonomy for direction of change.csv')
    change_dict = {key: [value for value in change_dict[key] if re.search(r'[a-zA-Z0-9]+', value)] for key in change_dict.keys()}
    all_topics = csv_to_dictionary_semicolon_delimiter('all_topics.csv')
    all_topics = {key: [value for value in all_topics[key] if re.search(r'[a-zA-Z0-9]+', value)] for key in all_topics.keys()}
    all_topics = {k: [x.strip() for x in v] for k, v in all_topics.items()}
    return change_dict, all_topics

def main():
    change_dict, all_topics = load_taxonomy()
    distance = 3
    text_analysis("./Test", distance, all_topics, change_dict)

if __name__ == "__main__":
    main()

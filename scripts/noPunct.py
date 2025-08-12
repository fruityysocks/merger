

import pandas as pd
import re

def strip_trailing_punctuation(df, column_name):
    pattern = re.compile(r'[\.,;:!?]+$')

    df = df.copy()

    def remove_punct(text):
        if isinstance(text, str):
            return pattern.sub('', text)
        return text  

    df[column_name] = df[column_name].apply(remove_punct)

    return df


df = pd.read_csv("data/exp1/csvs/day10Vowels.csv")
print(df.columns.tolist()) 

print("Before:")
print(df)

df_clean = strip_trailing_punctuation(df, 'word')

print("\nAfter:")
print(df_clean)

output = 'data/exp1/csvs/day10VowelsClean.csv'
df_clean.to_csv(output, index=False)

df = pd.read_csv("data/exp1/csvs/day10VowelsClean.csv")
df_sorted = df.sort_values(by=['speaker1_id', 'speaker', 'vowel'])
output = 'data/exp1/csvs/day10VowelsClean.csv'
df_sorted.to_csv(output, index=False)
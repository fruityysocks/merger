import pandas as pd

tokens_df = pd.read_csv('data/vowels.csv')

mean_formants = tokens_df.groupby(['speaker', 'vowel'])[['F1_lobnorm', 'F2_lobnorm']].mean().reset_index()

output_file = 'data/speakerVowelMeans.csv'
mean_formants.to_csv(output_file, index=False)

print(f"Saved mean formants per speaker-vowel pair to '{output_file}'")
import pandas as pd

def combine_birthyear(df1, df2, output_file, id_col='speaker', birthyear_col='birthyear'):
    # Extract unique speaker:birthyear pairs from df1
    speaker_birthyear_map = df1[[id_col, birthyear_col]].drop_duplicates().set_index(id_col)[birthyear_col]

    # Map birthyear for each speaker in df2
    df2[birthyear_col] = df2[id_col].map(speaker_birthyear_map)

    # Convert birthyear to nullable integer (allows NaN)
    df2[birthyear_col] = df2[birthyear_col].astype('Int64')

    # Save to CSV
    df2.to_csv(output_file, index=False)
    print(f"Combined file saved as {output_file}")

if __name__ == "__main__":
    df1 = pd.read_csv('data/vowels.csv')         
    df2 = pd.read_csv('data/demographics.csv')   

    output = 'data/updatedDemographics.csv'

    combine_birthyear(df1, df2, output)
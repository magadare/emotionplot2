import pandas as pd
from nrclex import NRCLex

def extract_emotional_words(df, sentence_column="Sentence"):
    """
    Extracts emotional words from the specified column in a DataFrame using NRCLex.

    Parameters:
    - df (pd.DataFrame): DataFrame containing sentences.
    - sentence_column (str): Column name containing sentences (default: 'Sentence').

    Returns:
    - pd.DataFrame: Updated DataFrame with a new 'words' column containing emotional words.
    """
    # Function to extract emotional words from a sentence
    def extract_emotional_words(sentence):
        emotion_obj = NRCLex(str(sentence))  # Process sentence with NRCLex
        return " ".join(emotion_obj.affect_list)  # Extract emotional words

    # Apply extraction to each sentence
    df["words"] = df[sentence_column].apply(extract_emotional_words)

    return df

# Example usage:
# df = extract_emotional_words_from_df(df)
# print(df.head())

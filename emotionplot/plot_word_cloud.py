import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

def generate_wordclouds(df, chunk_size):
    """
    Generates word clouds for grouped sentences.

    Parameters:
    - df (pd.DataFrame): DataFrame containing sentences and emotional words.
    - chunk_size (int): Number of sentences per group.

    Returns:
    - Displays word clouds for each sentence group.
    """

    # Assign sentence groups dynamically based on chunk size
    df["Sentence_Group"] = df.index // chunk_size

    # Group emotional words by sentence group
    grouped_words = df.groupby("Sentence_Group")["words"].apply(lambda x: " ".join(x)).reset_index()

    # Create word clouds for each sentence group
    fig, axes = plt.subplots(1, len(grouped_words), figsize=(15, 6))

    # Generate word cloud for each sentence group
    for i, row in grouped_words.iterrows():
        wordcloud = WordCloud(width=400, height=400, background_color="white").generate(row["words"])
        axes[i].imshow(wordcloud, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(f"Chunk {row['Sentence_Group']}")

    # Show the word clouds
    plt.tight_layout()
    plt.show()

# Example usage:
# generate_wordclouds(df, 400)

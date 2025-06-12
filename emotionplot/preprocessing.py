import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def preprocessing(content):
    """
    Preprocesses the input text by:
    1. Lowercasing the text.
    2. Removing numbers.
    3. Removing punctuation.
    4. Tokenizing the text into words.
    Args:
        content (str): The input text to preprocess.
    Returns:
        str: The preprocessed text with words separated by spaces.
    """
    # Lowercase
    content = content.lower()

    # Remove numbers
    content = ''.join([char for char in content if not char.isdigit()])

    # Remove punctuation
    punctuation = ['!','"','#','$','%','&','(',')',
                   '*','+','-','/',':',';','<',
                   '=','>','@',"\\",'"]"','^','_']


    for punct in punctuation:
        content = content.replace(punct, ' ')

    # Tokenize (split into words)
    tokens = content.split()

    return ' '.join(tokens)


def chunk_by_sentences(content, sentences_per_chunk=3):
    sentences = sent_tokenize(content)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = sentences[i:i+sentences_per_chunk]
        chunks.append(" ".join(chunk))
    df = pd.DataFrame({'chunk': chunks})
    return df

# chunks = chunk_by_sentences(data, sentences_per_chunk=10)

# Create DataFrame with each chunk as a row
# df = pd.DataFrame({'chunk': chunks})

# df["cleaned_chunk"] = df["chunk"].apply(preprocessing)

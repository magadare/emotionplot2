import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

def preprocessing(text):
    # Lowercase
    text = text.lower()

    # Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])

    # Remove punctuation
    punctuation = ['!','"','#','$','%','&','(',')',
                   '*','+','-','/',':',';','<', 
                   '=','>','@',"\\",'"]"','^','_']     
    

    for punct in punctuation:
        text = text.replace(punct, ' ')

    # Tokenize (split into words)
    tokens = text.split()

    return ' '.join(tokens)


def chunk_by_sentences(text, sentences_per_chunk=10):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = sentences[i:i+sentences_per_chunk]
        chunks.append(" ".join(chunk))

    return chunks

# chunks = chunk_by_sentences(data, sentences_per_chunk=10)

# Create DataFrame with each chunk as a row
# df = pd.DataFrame({'chunk': chunks})

# df["cleaned_chunk"] = df["chunk"].apply(preprocessing)
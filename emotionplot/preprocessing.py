import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re

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



def latex_to_paragraph_dataframe(latex_text):
    """
    Parses LaTeX-formatted text, groups lines into paragraphs based on single \newline breaks,
    and stores each paragraph in a DataFrame row.
    :param latex_text: LaTeX-formatted string.
    :return: Pandas DataFrame with paragraphs as rows.
    """
    # Remove LaTeX document structure
    latex_text = re.sub(r"\\documentclass{.*?}|\\begin{document}|\\end{document}", "", latex_text, flags=re.DOTALL)
    # Split text by isolated \newline (i.e., it appears on a line by itself)
    raw_paragraphs = re.split(r"\s*\n\s*\\newline\s*\n\s*", latex_text.strip())
    # Merge paragraph lines (inside each paragraph) into a single text block
    paragraphs = [" ".join(re.split(r"\s*\\newline\s*", para)).strip() for para in raw_paragraphs]
    # Remove remaining LaTeX commands and extra spaces
    paragraphs = [re.sub(r"\\[a-zA-Z]+", "", para).strip() for para in paragraphs if para.strip()]
    # Convert to DataFrame
    df = pd.DataFrame({"Paragraph": paragraphs})
    return df


def create_line_chunks(lines: list) -> pd.DataFrame:
    """
    Create individual chunks for each line of the poem.
    Args:
        lines (list): List of preprocessed poem lines
    Returns:
        pd.DataFrame: DataFrame with each line as a separate chunk
    """
    chunks = []

    for i, line in enumerate(lines):
        if line.strip():  # Only process non-empty lines
            chunks.append({
                'chunk': line,  # Keep 'chunk' column name for compatibility with predict_emotions
                'line_number': i + 1,
                'line_text': line
            })

    return pd.DataFrame(chunks)

def chunk_by_lines(lines: list, lines_per_chunk: int) -> pd.DataFrame:
    """
    Chunk poem lines into groups for emotion analysis.
    Args:
        lines (list): List of poem lines
        lines_per_chunk (int): Number of lines per chunk
    Returns:
        pd.DataFrame: DataFrame with chunks
    """
    chunks = []

    for i in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[i:i + lines_per_chunk]
        chunk_text = ' '.join(chunk_lines)  # Join lines with space for analysis
        chunks.append({
            'chunk_id': i // lines_per_chunk,
            'chunk': chunk_text,
            'line_numbers': f"{i+1}-{min(i+lines_per_chunk, len(lines))}",
            'original_lines': chunk_lines
        })

    return pd.DataFrame(chunks)

from bs4 import BeautifulSoup
import requests
import re

def get_novel(url: str) -> str:
    """
    Fetches the text of a novel from Project Gutenberg.

    Args:
        url (str): The URL of the novel on Project Gutenberg.

    Returns:
        str: The cleaned text of the novel.
    """
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, "html.parser")

    # Extract plain text from the BeautifulSoup object
    raw_text = soup.get_text()

    # Clean the Gutenberg text
    return raw_text



def clean_gutenberg_text(raw_text: str) -> str:
    # Remove BOM if present
    raw_text = raw_text.lstrip('\ufeff')

    # Markers to locate main content
    start_marker = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK"
    end_marker   = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK"

    start_match = re.search(start_marker, raw_text, re.IGNORECASE)
    end_match   = re.search(end_marker, raw_text, re.IGNORECASE)

    if start_match and end_match:
        content = raw_text[start_match.end():end_match.start()]
    else:
        # Fallback if markers not found
        content = raw_text

    # Normalize line endings and whitespace
    content = re.sub(r'\r\n', '\n', content)              # unify line endings
    content = re.sub(r'\n{2,}', '\n\n', content)          # collapse multiple blank lines
    content = re.sub(r'[ \t]+', ' ', content)             # collapse spaces/tabs
    content = content.strip()

    return content

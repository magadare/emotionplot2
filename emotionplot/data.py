from bs4 import BeautifulSoup
import requests
import re

def get_novel(url: str) -> str:
    """
    Fetches the raw text of a novel from Project Gutenberg.
    Automatically handles conversion from HTML URL to raw .txt format.

    Args:
        url (str): The Project Gutenberg URL (HTML or .txt).

    Returns:
        str: The raw text of the novel.
    """
    # Convert HTML-style Gutenberg book URL to raw .txt format
    if "gutenberg.org/ebooks/" in url:
        book_id = url.rstrip("/").split("/")[-1]
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

    # Fetch the content
    resp = requests.get(url)
    resp.raise_for_status()  # Raise an error if the request failed

    # Handle content type
    content_type = resp.headers.get("Content-Type", "")
    if "html" in content_type:
        soup = BeautifulSoup(resp.content, "html.parser")
        raw_text = soup.get_text()
    else:
        raw_text = resp.text

    return raw_text




def clean_gutenberg_text(raw_text: str) -> str:

    """
    Cleans the raw text from Project Gutenberg by removing headers, footers,
    and other non-content elements, and normalizing whitespace.
    Args:
        raw_text (str): The raw text from Project Gutenberg.
    Returns:
        str: The cleaned text, ready for analysis.
    """

    # Ensure the input is a string
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

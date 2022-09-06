
import re
from typing import Optional, Tuple


def remove_abstract_references(text: str, max_intro_idx: Optional[int] = 6000) -> Tuple[str, bool, bool]: 
    """Try to delete the abstract and references 
    of a text.
    Parameters: 
        text (str): input text
        max_intro_idx (int): maximum index where matches for "Introduction" 
            can be found. Default to 6000 characters.

    Returns: 
        text_trimmed (str): Trimmed text without abstract and references
        abs_trimmed (bool): Boolean indicating whether the abstract was trimmed
        ref_trimmed (bool): Boolean indicating whether the references were trimmed
    """

    intro_idx = re.search(r"Introduction|INTRODUCTION", text[:max_intro_idx])
    abs_trimmed = False if not intro_idx else True
    intro_idx = 0 if not intro_idx else intro_idx.end()

    ref_idx = re.search(r"References\n", text)
    ref_trimmed = False if not ref_idx else True
    ref_idx = len(text) if not ref_idx else ref_idx.start()

    text_trimmed = text[intro_idx:ref_idx]

    return (text_trimmed, abs_trimmed, ref_trimmed)

def clean_body_text(text: str) -> str: 
    """
    Cleans a body of text before further processing.
    """

    # Strip initial and final whitespaces
    text = text.strip()

    # Remove lines that have less than 5 charaters
    text = re.sub(r"\n(.{1,6})\n","\n\n", text)
    text = re.sub(r"\n(.{1,6})\n","", text)

    # Collapse multiple newlines into one
    text = re.sub(r'\n+', r'\n', text)

    # Remove hyphenation
    text = re.sub(r'-\n', '', text)

    # Add dot at the end of equations
    text = re.sub(r'(\(\d+\))', r'\1.', text)

    # Substitute blank lines if not preceded by a period by a whitespace
    text = re.sub(r'([^\.])\n', r'\1 ', text)

    # Substitute blank lines if preceded by a period by a whitespace
    text = re.sub(r'([\.])\n', r'\1 ', text)
    
    # Lowercase everything
    text = text.lower()
    
    # Remove the final blank lines that might still remain
    text = re.sub(r'\n', '', text)
    
    # Remove everything between parenthesis (mostly citations)
    text = re.sub(r' \([^()]*\)', '', text) 

    # Remove links
    text = re.sub(r'http[s]?://\S+', '', text)

    # Get right quotation marks symbols
    text = re.sub(r"“|”", r'"', text)
    text = re.sub(r"’", r"'", text)

    return text

def remove_short_sentences(text):
    """Remove sentences with less than 10 words"""
    list_sent = text.split('. ')
    return ". ".join([sent for sent in list_sent if len(sent.split()) > 10])
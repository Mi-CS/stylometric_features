import re
from typing import Optional, Tuple
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

def clean_text(text: str,
               ignore_trimmed_fail: bool = False) -> str:

    """
    Pre-process a text parsed from an article. 
    Parameters: 
        text (str): parsed article text as string
        ignore_trimmed_fail (bool) = False: whether to return 
            text if no intro and references were trimmed. By default
            returns an empty string if either removing the introduction
            or the references failed.
    Returns: 
        text (str): string containing the text after the preprocessing
            steps
    """
    
    text, int_rm, ref_rm = _remove_abstract_references(text)
    if not ignore_trimmed_fail and (not int_rm or not ref_rm):
        return ""
    text = _clean_body_text(text)
    text = _remove_short_sentences(text)
    text = _clean_digit_sentences(text)
    text = _clean_tokens(text)
    return text


def _remove_abstract_references(text: str, max_intro_idx: Optional[int] = 6000) -> Tuple[str, bool, bool]: 
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

def _clean_body_text(text: str) -> str: 
    """
    Cleans a body of text at a character level.
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

def _remove_short_sentences(text: str) -> str:
    """Remove sentences with less than 10 words"""
    list_sent = text.split('. ')
    return ". ".join([sent for sent in list_sent if len(sent.split()) > 10])


def _clean_digit_sentences(text: str) -> str: 
    """Remove sentences whose tokens consist of more than
    30% of digits"""

    def has_too_many_digits(sentence: str) -> bool: 
        n_tokens = len(re.findall(r"[\w]+", sentence))
        n_digits = len(re.findall(r"[0-9]+", sentence))
        if not n_tokens:
            return True
        if n_digits / n_tokens > 0.3: 
            return True
        return False


    sentences = [sentence for sentence in sent_tokenize(text) if not
                        has_too_many_digits(sentence)]
    
    return " ".join(sentences)



def _clean_tokens(text: str,
                 allowed_special_chars: Optional[str] = None) -> str: 
    """
    Clean a body of text at a token level.
    """

    if not allowed_special_chars:
        allowed_special_chars = r"(),.:;-–!?_'" + '"'

    # List of allowed tokens
    tokens = [token for token in word_tokenize(text) if 
              (token in allowed_special_chars) or 
              (re.findall(r"[a-z]+", token))]
    
    # Detokenize using Treebank
    detoken = TreebankWordDetokenizer()
    text = detoken.detokenize(tokens)

    # Correct blank spaces between last word and final dot
    text = re.sub(r"(\w) \. (\w)", r"\1. \2", text)

    return text

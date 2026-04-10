"""Utility functions for extracting answers from model responses.

This module provides functions to extract answers from various model response formats,
including support for thinking models that use <answer> tags.

The extraction behavior can be controlled via the LMMS_EXTRACT_ANSWER_FROM_TAGS
environment variable:
    - "auto": Automatically detect and extract from <answer> tags (default)
    - "true" or "1": Always extract from <answer> tags if present
    - "false" or "0": Never extract from <answer> tags, return raw response
"""

import os
import re
from typing import Optional


# Environment variable to control answer tag extraction
_EXTRACT_FROM_TAGS_ENV = "LMMS_EXTRACT_ANSWER_FROM_TAGS"


def should_extract_from_tags() -> bool:
    """Check if answer extraction from <answer> tags should be performed.

    Reads the LMMS_EXTRACT_ANSWER_FROM_TAGS environment variable:
    - "auto": Detect automatically (default) - extracts if <answer> tags present
    - "true"/"1": Always extract from <answer> tags
    - "false"/"0": Never extract, return raw response

    Returns:
        True if extraction should be performed, False otherwise.
    """
    env_value = os.environ.get(_EXTRACT_FROM_TAGS_ENV, "").lower()

    if env_value == "auto":
        return True  # Auto-detect: always try to extract
    elif env_value in ("true", "1", "yes"):
        return True
    elif env_value in ("false", "0", "no"):
        print(f"LMMS: Skipping <answer> tag extraction as {_EXTRACT_FROM_TAGS_ENV} is set to '{env_value}'")
        return False
    else:
        return True  # Default to auto-detect


def extract_answer(response: str, use_answer_tag: Optional[bool] = None) -> str:
    """Extract answer from model response, with support for <answer> tags.

    This function handles:
    1. Responses with <answer>...</answer> tags (from thinking models)
    2. Plain text responses (traditional models)

    The behavior can be controlled via:
    - The use_answer_tag parameter (overrides environment variable)
    - The LMMS_EXTRACT_ANSWER_FROM_TAGS environment variable

    Args:
        response: The model's response string
        use_answer_tag: If True, try to extract from <answer> tags first.
                       If False, return the original response stripped.
                       If None (default), use environment variable setting.

    Returns:
        The extracted answer string.

    Examples:
        >>> extract_answer("<answer>yes</answer>")
        'yes'
        >>> extract_answer("The answer is yes")
        'The answer is yes'
        >>> extract_answer("<think_off>\\n</think_off>\\n<answer>no</answer>")
        'no'
    """
    if not response:
        return ""

    response = response.strip()

    # Determine if we should extract from answer tags
    extract = (
        use_answer_tag if use_answer_tag is not None else should_extract_from_tags()
    )

    print(f"LMMS: Extracting answer from response with use_answer_tag={extract}")
    
    # Try to extract content from <answer> tags if enabled
    if extract:
        # Use DOTALL flag to match across newlines
        # \\
        # box_match = re.search
        # if box_match:
        #     return box_match.group(1).strip()
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        

    # Return the original response if no <answer> tag found or extraction disabled
    return response


def extract_answer_lowercase(response: str) -> str:
    """Extract answer and convert to lowercase for comparison.

    Args:
        response: The model's response string

    Returns:
        The extracted answer in lowercase.
    """
    return extract_answer(response).lower()


def extract_answer_multi_choice(response: str, choices: list) -> str:
    """Extract answer from multiple choice response.

    First tries to extract from <answer> tags, then falls back to
    standard multiple choice parsing.

    Args:
        response: The model's response string
        choices: List of possible choices (e.g., ['A', 'B', 'C', 'D'])

    Returns:
        The extracted choice letter.
    """
    extracted = extract_answer(response)

    # If extracted content is a single letter that matches a choice, return it
    if len(extracted) == 1 and extracted.upper() in choices:
        return extracted.upper()

    # Otherwise, search for choice letters in the extracted content
    for choice in choices:
        if (
            f"({choice})" in extracted
            or f"{choice}." in extracted
            or f" {choice} " in extracted
        ):
            return choice

    # If no choice found, return the extracted content as-is
    return extracted

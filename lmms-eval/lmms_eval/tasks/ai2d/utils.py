import re
import os

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter
from lmms_eval.tasks._task_utils.answer_extraction import extract_answer_lowercase


def ai2d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]

    if os.environ.get("AI2D_DOC_TO_TEXT_DEBUG", "0") == "1":
        profile = None
        if isinstance(lmms_eval_specific_kwargs, dict):
            profile = lmms_eval_specific_kwargs.get("profile")
        print(f"[ai2d] using lmms_eval_specific_kwargs profile: {profile}")
        
    if lmms_eval_specific_kwargs["prompt_format"] == "mcq":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "qa":
        options = "\n".join(choices)
        return f"{pre_prompt}{question}{options}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "mcq_xcomposer":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = " ".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\nContext: N/A\n{choices_str}{post_prompt}"
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs['prompt_format']}")


def ai2d_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ai2d_doc_to_target(doc, model_specific_target_kwargs):
    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        return doc["options"][int(doc["answer"])]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex patterns
            answer_tag_regex = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.IGNORECASE)
            option_letter_regex = re.compile(r"^\s*([A-Z])(?:\.|:|\)|$)", re.IGNORECASE)  # Start with A. or A: or A) or just A
            text_regex = re.compile(r"(?:answer|option|choice)(?: is)?\s*:?\s*([A-Z])", re.IGNORECASE)

            # Process each response
            filtered = []
            for resp in r:
                extracted = None

                # 1. Try to extract from <answer> tags (for thinking models)
                answer_match = answer_tag_regex.search(resp)
                if answer_match:
                    extracted = answer_match.group(1).upper()
                
                # 2. Try to match the option letter at the start of the response
                if not extracted:
                    match = option_letter_regex.match(resp)
                    if match:
                        extracted = match.group(1).upper()
                
                # 3. Try to match "Answer is A" pattern
                if not extracted:
                    match = text_regex.search(resp)
                    if match:
                        extracted = match.group(1).upper()

                if extracted:
                    filtered.append(extracted)
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps


def ai2d_process_results(doc, results):
    """Process results using extract_answer_lowercase to extract answer from tags."""
    pred = results[0].strip()
    gt_idx = int(doc["answer"])
    len_choices = len(doc["options"])
    options = [chr(ord("A") + i) for i in range(len_choices)]

    # Normalize prediction for comparison - should be a letter like A, B, C, D
    pred_normalized = pred.strip().upper()
    if len(pred_normalized) > 0 and pred_normalized[0] in options:
        pred_normalized = pred_normalized[0]

    # Ground truth is an index (0, 1, 2, 3), convert to letter for comparison
    gt_letter = options[gt_idx]

    # Check if prediction matches ground truth
    score = 1 if pred_normalized == gt_letter else 0

    return {
        "exact_match": {
            "score": score,
            "prediction": results[0],
            "ground_truth": gt_letter,
        },
    }


def ai2d_aggregate_accuracy(results):
    """Aggregate results by computing mean accuracy."""
    total = len(results)
    correct = sum(result["score"] for result in results)
    return correct / total if total > 0 else 0

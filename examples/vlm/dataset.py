"""
Dataset implementation for Visual Language Models (VLM).
Handles lazy loading of JSONL data and image path resolution.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from torch.utils.data import Dataset


class LazyVLMJsonlDataset(Dataset):
    """
    A lazy-loading dataset for VLM tasks using JSONL format.
    
    This dataset does not load all data into memory. Instead, it builds an index of 
    file offsets to read samples on-the-fly. It supports flexible image path resolution
    and integrates with tokenizer chat templates.

    Expected JSONL format per line:
    {
        "id": "...",
        "image": "path/to/img.jpg" or ["path/to/img.jpg"],
        "messages": [
            {"role": "user", "content": "..."},
            ...
        ],
        "gt_answer": "..."
    }
    """

    def __init__(
        self,
        data_path: str,
        processor: Any,
        max_length: int = 2048,
        base_image_path: Optional[str] = None,
        print_example: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the .jsonl file.
            processor: HuggingFace processor or tokenizer object.
            max_length: Maximum sequence length (reserved for future use).
            base_image_path: Optional root directory for relative image paths.
            print_example: If True, prints a processed example during initialization.
        """
        self.data_path = data_path
        self.processor = processor
        self.max_length = max_length
        self.base_dir = os.path.dirname(data_path)
        self.base_image_path = base_image_path
        self.offsets = [0]

        print(f"Indexing dataset (Lazy Loading): {data_path} ...")
        self._index_file()
        print(f"Indexed {len(self.offsets)} samples.")

        if print_example and len(self.offsets) > 0:
            self._print_and_save_example()

    def _index_file(self) -> None:
        """Builds file offsets for random access."""
        with open(self.data_path, "rb") as f:
            while f.readline():
                self.offsets.append(f.tell())
        # Remove the last offset which points to EOF
        if self.offsets:
            self.offsets.pop()

    def __len__(self) -> int:
        return len(self.offsets)

    def _resolve_image_path(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Resolves the absolute path for the image in the sample.

        Priority:
        1. 'image' field
        2. 'tos_key' field (legacy/specific support)

        Resolution Logic:
        1. Absolute paths are returned as-is.
        2. If base_image_path is set, join it with the relative path.
        3. Fallback: try relative to the dataset file location.
        4. Fallback: try relative to the current working directory.
        """
        # 1. Extract image path from sample
        image_rel_path = sample.get("image")
        if not image_rel_path:
            # Fallback for datasets using 'tos_key'
            tos_key = sample.get("tos_key")
            if tos_key:
                image_rel_path = tos_key[0] if isinstance(tos_key, list) else tos_key
        
        if not image_rel_path:
            return None

        if isinstance(image_rel_path, list):
            image_rel_path = image_rel_path[0]

        # 2. Check if path is already absolute
        if os.path.isabs(image_rel_path):
            if self.base_image_path:
                print(f"Warning: Image path is already absolute, ignoring base_image_path: {image_rel_path}")
            return image_rel_path

        # 3. Try resolving with base_image_path
        if self.base_image_path:
            potential_path = os.path.join(self.base_image_path, image_rel_path)
            if os.path.exists(potential_path):
                return potential_path
            print(f"Warning: Image not found at constructed path: {potential_path}")
            return None

        # 4. Fallback: Try relative to dataset directory
        potential_path = os.path.join(self.base_dir, image_rel_path)
        if os.path.exists(potential_path):
            return potential_path

        # 5. Fallback: Try relative to CWD
        cwd_path = os.path.abspath(image_rel_path)
        if os.path.exists(cwd_path):
            return cwd_path

        return None

    def _format_messages(self, sample: Dict[str, Any]) -> str:
        """
        Constructs the prompt string using the processor's chat template.
        
        Handles:
        - Parsing user messages.
        - Splitting <image> tags into multimodal content.
        - Appending optional instructions.
        """
        user_content = ""
        user_append = ""

        # Extract user content and optional append instruction
        if "messages" in sample:
            for message in sample["messages"]:
                if message.get("role") == "user":
                    val = message.get("content", "")
                    user_content = val
                    user_append = message.get("append", "")
                    break
        
        # Build multimodal messages structure
        messages = []
        if isinstance(user_content, str):
            if "<image>" in user_content:
                parts = user_content.split("<image>")
                content_parts = []
                for i, part in enumerate(parts):
                    if part:
                        content_parts.append({"type": "text", "text": part})
                    if i < len(parts) - 1:
                        content_parts.append({"type": "image"})
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": user_content}]})
        elif isinstance(user_content, list):
            # Already in multimodal format
            messages.append({"role": "user", "content": user_content})
        else:
            # Fallback for other types
            messages.append({"role": "user", "content": [{"type": "text", "text": str(user_content)}]})

        # Apply chat template
        # Handle cases where processor wraps tokenizer or is the tokenizer
        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                base_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Warning: apply_chat_template failed: {e}. Using plain text fallback.")
                base_prompt = self._fallback_text_extraction(messages)
        else:
            print("Warning: Tokenizer missing 'apply_chat_template'. Using plain text fallback.")
            base_prompt = self._fallback_text_extraction(messages)
        
        return base_prompt + user_append

    def _fallback_text_extraction(self, messages: List[Dict[str, Any]]) -> str:
        """Extracts plain text from messages when template application fails."""
        text = ""
        for part in messages[0]["content"]:
            if part.get("type") == "text":
                text += part["text"]
        return text

    def _get_ground_truth(self, sample: Dict[str, Any]) -> str:
        """Extracts the ground truth answer from the sample."""
        raw_answer = sample.get("gt_answer", "")
        if not raw_answer:
            # You might want to raise an error or return None depending on strictness
            # raise ValueError(f"Sample {sample.get('id', 'unknown')} has no gt_answer!")
            pass
        return raw_answer

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves and processes a sample by index.
        Recursively tries the next sample if the current one is invalid.
        """
        offset = self.offsets[index]
        
        # Lazy load: read specific line from file
        with open(self.data_path, "rb") as f:
            f.seek(offset)
            line_bytes = f.readline()
            
        try:
            line = line_bytes.decode("utf-8")
            sample = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(f"Warning: Failed to decode/parse line at index {index}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        # Validate ground truth
        raw_answer = self._get_ground_truth(sample)
        if not raw_answer:
             # If strict validation is needed, uncomment next line
             # print(f"Warning: Sample {index} missing answer. Skipping.")
             # return self.__getitem__((index + 1) % len(self))
             pass

        # Resolve and load image
        image_path = self._resolve_image_path(sample)
        if image_path is None:
            print(f"Warning: Image path missing for sample {index}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        try:
            image_obj = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to open image {image_path}: {e}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        # Format prompt
        messages_text = self._format_messages(sample)

        return {
            "messages": messages_text,
            "images": image_obj,
            "answer": raw_answer,
            "query_id": sample.get("query_id", sample.get("id", str(index)))
        }

    def _print_and_save_example(self) -> None:
        """Prints a processed example to stdout for verification."""
        print("\n--- Dataset Example ---")
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                line = f.readline()
                if not line:
                    print("Empty file.")
                    return
                sample = json.loads(line)
        except Exception as e:
            print(f"Could not load example: {e}")
            return

        print(f"Raw sample keys: {list(sample.keys())}")
        
        processed_prompt = self._format_messages(sample)
        gt_answer = self._get_ground_truth(sample)
        
        print(f"\nProcessed Prompt:\n{repr(processed_prompt)}")
        print(f"\nGround Truth Answer: {repr(gt_answer)}")
        print("--- End Example ---\n")

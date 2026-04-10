import json
import os

import pandas as pd
from datasets import Dataset, Features, Image, Sequence, Value
from tqdm import tqdm

# Define the features for the dataset
features = Features(
    {
        "video": Value(dtype="string"),
        "caption": Value(dtype="string"),
        "timestamp": Sequence(Value(dtype="float16")),  # Use Sequence for lists
    }
)

df_items = {
    "video": [],
    "caption": [],
    "timestamp": [],
}

# Load json file
json_path = "/opt/tiger/lmms-eval/lmms_eval/tasks/charades_sta/temporal_grounding_charades.json"
with open(json_path, "r") as f:
    data = json.load(f)


# Iterate over the rows of the data
for cur_meta in data:
    video = cur_meta["video"]
    caption = cur_meta["caption"]
    timestamp = cur_meta["timestamp"]
    # import pdb;pdb.set_trace()
    df_items["video"].append(video)
    df_items["caption"].append(caption)
    df_items["timestamp"].append(timestamp)

import pdb

pdb.set_trace()

df_items = pd.DataFrame(df_items)

dataset = Dataset.from_pandas(df_items, features=features)

hub_dataset_path = "lmms-lab/charades_sta"
dataset.push_to_hub(repo_id=hub_dataset_path, split="test")
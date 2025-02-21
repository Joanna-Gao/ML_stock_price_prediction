from typing import List
import json
import os
import numpy as np


def save_meta_data(
    output_dir: str,
    ticker: str,
    time_period: List[str],
    look_back: int,
    batch_size: int,
    model_config: dict,
):
    meta_data = {
        "ticker": ticker,
        "time_period": time_period,
        "look_back": look_back,
        "batch_size": batch_size,
        "model_config": model_config,
    }

    with open(os.path.join(output_dir, "meta_data.json"), "w") as f:
        json.dump(meta_data, f, indent=4)

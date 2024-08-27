import argparse
import os
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from delia.utils import (
    CustomSFTTrainer,
    load_and_process_datasets,
)

def DSFTTrainer(cache_dir = None, diverse_ratio = 100, train_dataset = None, eval_dataset = None, **SFTTrainerArgs):
    """
    Trains a Deep Sparse Fine-Tuning (DSFT) model.

    Args:
        cache_dir (str): The directory path where the cached datasets are stored. Defaults to None.
        diverse_ratio (int): The ratio of diverse samples to be included in the combined dataset. Defaults to 100.
        train_dataset (Dataset): The directory path for training dataset. Defaults to None.
        eval_dataset (Dataset): The directory path for evaluation dataset. Defaults to None.
        **SFTTrainerArgs: Additional arguments to be passed to the CustomSFTTrainer.

    Returns:
        CustomSFTTrainer: The trained DSFT model trainer.
    
    Raises:
        ValueError: If cache_dir is provided but does not exist.
        ValueError: If cache_dir is not provided.
    """
    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            raise ValueError("cache_dir does not exist")
        combined_dataset, val_dataset = load_and_process_datasets(
            cache_dir, train_dataset, eval_dataset, diverse_ratio
        )
    else:
        raise ValueError("cache_dir must be provided")
    trainer = CustomSFTTrainer(**SFTTrainerArgs, train_dataset=combined_dataset, eval_dataset=val_dataset)
    return trainer
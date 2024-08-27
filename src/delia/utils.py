import os
import json
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import inspect

class CustomSFTTrainer(SFTTrainer):
    """
    This class extends the functionality of the SFTTrainer to implement DELIA by adding additional methods and attributes.
    
    This class adds methods for tracking and analyzing important token losses during evaluation,
    which is particularly useful for understanding model performance on specific tokens or patterns.

    Attributes:
        eval_important_loss (dict): Stores important token losses during evaluation.
    """
    @classmethod
    def from_sft_trainer(cls, sft_trainer, **kwargs):
        """
        Create a CustomSFTTrainer instance from an existing SFTTrainer.

        This method allows for easy conversion from a standard SFTTrainer to a CustomSFTTrainer,
        preserving all relevant parameters and adding any additional kwargs.

        Args:
            sft_trainer (SFTTrainer): An instance of SFTTrainer to convert.
            **kwargs: Additional keyword arguments to override or add to the trainer configuration.

        Returns:
            CustomSFTTrainer: A new instance of CustomSFTTrainer.
        """
        sft_init_params = inspect.signature(SFTTrainer.__init__).parameters
        sft_params = {}
        for param_name in sft_init_params:
            if param_name != 'self' and hasattr(sft_trainer, param_name):
                sft_params[param_name] = getattr(sft_trainer, param_name)
        sft_params.update(kwargs)
        print(sft_params)
        return cls(**sft_params)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_important_loss = {}

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if self.args.do_eval:
            with torch.no_grad():
                self.eval_important_loss[str(self.state.global_step)] = []
                for i in range(inputs["input_ids"].shape[0]):
                    sample_input_ids = inputs["input_ids"][i]
                    sample_attention_mask = inputs["attention_mask"][i]
                    sample_labels = inputs["labels"][i]
                    sample_logits = outputs.logits[i]

                    valid_logits = sample_logits[sample_attention_mask.bool()]
                    valid_tokens = sample_input_ids[sample_attention_mask.bool()]

                    for j, token in enumerate(valid_tokens):
                        token_str = self.tokenizer.decode([token.item()])
                        if "'" in token_str or "<sep>" in token_str:
                            logit = valid_logits[j-1] if j > 0 else valid_logits[j]
                            label = sample_labels[j].item()
                            logit = logit.detach().cpu()
                            token_loss = F.cross_entropy(logit.unsqueeze(0), torch.tensor([label], device=logit.device)) 

                            self.eval_important_loss[str(self.state.global_step)].append({
                                "token": token_str,
                                "loss": token_loss.item(),
                            })

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        self.eval_important_loss = {}
        metrics = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        with open("eval_important_loss.jsonl", "a+") as f:
            json.dump(self.eval_important_loss, f)
            f.write('\n')
            
        return metrics

# utils.py

import random
from datasets import load_dataset, concatenate_datasets

def load_and_process_datasets(dataset1_path, dataset2_path, val_dataset_path, diverse_ratio=100.0):
    """
    Load and process datasets.
    Args:
        dataset1_path (str): Path to diverse dataset.
        dataset2_path (str): Path to downstream dataset.
        val_dataset_path (str): Path to validation dataset.
        diverse_ratio (float, optional): Ratio of diverse dataset to downstream dataset. Defaults to 100.0.
    Returns:
        tuple: A tuple containing the combined dataset and the validation dataset.
    """
    pass
    dataset2 = load_dataset("json", data_files=dataset2_path, split="train")
    dataset2 = dataset2.map(lambda example, idx: {**example, "dataset_source": "dataset2", "data_index": idx}, with_indices=True)

    if dataset1_path and diverse_ratio > 0:
        dataset1 = load_dataset("json", data_files=dataset1_path, split="train")
        dataset1 = dataset1.map(lambda example, idx: {**example, "dataset_source": "dataset1", "data_index": idx}, with_indices=True)
        num_samples_dataset2 = len(dataset2)
        num_samples_dataset1 = int(num_samples_dataset2 * diverse_ratio)
        if num_samples_dataset1 < len(dataset1):
            dataset1 = dataset1.shuffle(seed=42).select(range(num_samples_dataset1))
        
        combined_dataset = concatenate_datasets([dataset1, dataset2])
    else:
        combined_dataset = dataset2

    combined_dataset = combined_dataset.shuffle(seed=42)

    val_dataset = load_dataset("json", data_files=val_dataset_path, split="train")
    
    return combined_dataset, val_dataset


def setup_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_has_fp16_weight=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2", 
        quantization_config=quantization_config
        )

    special_tokens_dict = {"sep_token": "<sep>"}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens")
    assert tokenizer.sep_token == "<sep>"
    model.resize_token_embeddings((len(tokenizer)//64 + 1)*64)
    
    torch.manual_seed(2024)
    new_tokens_index = slice(len(tokenizer), (len(tokenizer) // 64 + 1) * 64)
    with torch.no_grad():
        embedding_dim = model.config.hidden_size
        new_embeddings = torch.randn((len(tokenizer) // 64 + 1) * 64 - len(tokenizer), embedding_dim)
        model.model.embed_tokens.weight[new_tokens_index] = new_embeddings

    return tokenizer, model

def get_peft_config():
    return LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "embed_tokens"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def get_training_arguments(output_dir, num_train_epochs, learning_rate, per_device_train_batch_size, gradient_accumulation_steps, max_seq_length):
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, 
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1,  
        save_steps=100, 
        warmup_steps=25, 
        lr_scheduler_type="constant_with_warmup",
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        max_seq_length=max_seq_length
    )
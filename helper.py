import sys
import os
import random
os.environ.pop("PYTHONPATH", None)  # Unset PYTHONPATH if it exists
# Filter out undesired directories
sys.path = [p for p in sys.path if not p.startswith("/opt/local/stow/pip-3.10")]
sys.modules.pop("typing_extensions", None)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset, Dataset

def get_device(gpu_id=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def load_model(model_name: str, num_labels: int, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model.to(device)

def load_data(dataset_name: str, split: str):
    return load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", dataset_name, split=split, trust_remote_code=True, streaming=False)

def preprocess_train_data(train_dataset, tokenizer, validation_split=0.05, random_state=42):
    """
    Preprocesses the training dataset by tokenizing the sequences and splitting into training and validation sets.
    
    Args:
        train_dataset: A HuggingFace Dataset object for training data containing 'sequence' and 'label'.
        tokenizer: The tokenizer to use.
        validation_split (float): Fraction of training data to use as validation.
        random_state (int): Seed for the train/validation split.
    
    Returns:
        tokenized_train: The tokenized training set.
        tokenized_validation: The tokenized validation set.
    """
    # Extract sequences and labels
    train_sequences = train_dataset['sequence']
    train_labels = train_dataset['label']
    
    # Split the training data into train and validation sets
    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(
        train_sequences, train_labels, test_size=validation_split, random_state=random_state)
    
    # Create HuggingFace Datasets
    ds_train = Dataset.from_dict({"data": train_sequences, "labels": train_labels})
    ds_validation = Dataset.from_dict({"data": validation_sequences, "labels": validation_labels})
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["data"], padding=True, truncation=True)
    
    # Map tokenization function to the datasets
    tokenized_train = ds_train.map(tokenize_function, batched=True, remove_columns=["data"])
    tokenized_validation = ds_validation.map(tokenize_function, batched=True, remove_columns=["data"])
    
    return tokenized_train, tokenized_validation

def preprocess_test_data(test_dataset, tokenizer):
    """
    Preprocesses the test dataset by tokenizing the sequences.
    
    Args:
        test_dataset: A HuggingFace Dataset object for test data containing 'sequence' and 'label'.
        tokenizer: The tokenizer to use.
    
    Returns:
        tokenized_test: The tokenized test set.
        test_labels: The original labels from the test dataset.
    """
    # Extract sequences and labels
    test_sequences = test_dataset['sequence']
    test_labels = test_dataset['label']
    
    # Create a HuggingFace Dataset for test data
    ds_test = Dataset.from_dict({"data": test_sequences, "labels": test_labels})
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["data"], padding=True, truncation=True)
    
    # Map tokenization function to the test dataset
    tokenized_test = ds_test.map(tokenize_function, batched=True, remove_columns=["data"])
    
    return tokenized_test, test_labels

def minimizer_tokenizer(seq, k, w, tokenizer):
    from minimizer import find_minimizers_custom
    minimizers = find_minimizers_custom(seq, k=k, w=w, canonical=True)
    canonical_seq = "".join(minimizers)
    return tokenizer(canonical_seq, padding=True, truncation=True)

def preprocess_train_data_with_minimizers(train_dataset, tokenizer, k, w, validation_split=0.05, random_state=42):
    train_sequences = train_dataset['sequence']
    train_labels = train_dataset['label']

    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(
        train_sequences, train_labels, test_size=validation_split, random_state=random_state)

    ds_train = Dataset.from_dict({"data": train_sequences, "labels": train_labels})
    ds_validation = Dataset.from_dict({"data": validation_sequences, "labels": validation_labels})

    def tokenize_function(examples):
        return minimizer_tokenizer(examples["data"], k, w, tokenizer)

    tokenized_train = ds_train.map(tokenize_function, batched=False, remove_columns=["data"])
    tokenized_validation = ds_validation.map(tokenize_function, batched=False, remove_columns=["data"])

    return tokenized_train, tokenized_validation

def preprocess_test_data_with_minimizers(test_dataset, tokenizer, k, w):
    test_sequences = test_dataset['sequence']
    test_labels = test_dataset['label']

    ds_test = Dataset.from_dict({"data": test_sequences, "labels": test_labels})

    def tokenize_function(examples):
        return minimizer_tokenizer(examples["data"], k, w, tokenizer)

    tokenized_test = ds_test.map(tokenize_function, batched=False, remove_columns=["data"])

    return tokenized_test, test_labels


def get_training_args(model_name: str, batch_size: int, num_labels: int):

    return TrainingArguments(
        output_dir=f"{model_name}-finetuned-NucleotideTransformer",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        logging_steps=100,
        load_best_model_at_end=True,  # Keep the best model according to thes evaluation
        metric_for_best_model="f1_score" if num_labels == 2 else "mcc_score",
        label_names=["labels"],
        dataloader_drop_last=False,
        max_steps=1000,
    )

# Define the metric for the evaluation
def compute_metrics_mcc(eval_pred):
    """Computes Matthews correlation coefficient (MCC score) for binary classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r={'mcc_score': matthews_corrcoef(references, predictions)}
    return r

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    return {"f1_score": f1_score(references, predictions), "mcc": matthews_corrcoef(references, predictions)}

def train_model(model, training_args, tokenized_train, tokenized_validation, tokenizer):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mcc if training_args.metric_for_best_model == "mcc_score" else compute_metrics,
    )
    trainer.train()
    return trainer

def save_model(trainer, model_name: str):
    trainer.model.save_pretrained(f"{model_name}-finetuned-NucleotideTransformer")
    trainer.tokenizer.save_pretrained(f"{model_name}-finetuned-NucleotideTransformer")

def load_saved_model(model_name: str, device):
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}-finetuned-NucleotideTransformer")
    tokenizer = AutoTokenizer.from_pretrained(f"{model_name}-finetuned-NucleotideTransformer")
    return model.to(device), tokenizer

def evaluate_model(trainer, tokenized_test, test_labels, model_name: str):
    test_results = trainer.predict(tokenized_test)
    predictions = np.argmax(test_results.predictions, axis=-1)
    cm = confusion_matrix(test_labels, predictions)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix on Test Set - {model_name}")
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.show()

    print("Test Metrics:")
    print(f"F1 Score: {test_results.metrics.get('test_f1_score', 'N/A')}")
    print(f"MCC: {matthews_corrcoef(test_labels, predictions)}")

def plot_f1_score(trainer, model_name: str):
    curve_evaluation_f1_score = [[a['step'], a['eval_f1_score']] for a in trainer.state.log_history if 'eval_f1_score' in a.keys()]
    eval_f1_score = [c[1] for c in curve_evaluation_f1_score]
    steps = [c[0] for c in curve_evaluation_f1_score]
    
    plt.figure()
    plt.plot(steps, eval_f1_score, 'b', label='Validation F1 score')
    plt.title(f'Validation F1 score for promoter prediction - {model_name}')
    plt.xlabel('Number of training steps performed')
    plt.ylabel('Validation F1 score')
    plt.legend()
    plt.savefig(f"validation_f1_score_{model_name}.png")
    plt.show()

def plot_mcc_score(trainer, model_name: str):
    curve_evaluation_mcc_score = [[a['step'], a['eval_mcc']] for a in trainer.state.log_history if 'eval_mcc' in a.keys()]
    eval_mcc_score = [c[1] for c in curve_evaluation_mcc_score]
    steps = [c[0] for c in curve_evaluation_mcc_score]
    
    plt.figure()
    plt.plot(steps, eval_mcc_score, 'b', label='Validation MCC score')
    plt.title(f'Validation MCC score for promoter prediction - {model_name}')
    plt.xlabel('Number of training steps performed')
    plt.ylabel('Validation MCC score')
    plt.legend()
    plt.savefig(f"validation_mcc_score_{model_name}.png")
    plt.show()

# ---- New functions for masked inference ----

def mask_sequence(token_ids, tokenizer, mask_ratio=0.2):
    """
    Randomly masks tokens in the sequence based on the given mask_ratio.
    If the model's tokenizer has no mask_token, uses the unk_token.
    
    Args:
        token_ids (list): List of token IDs.
        tokenizer: The tokenizer instance.
        mask_ratio (float): Ratio of tokens to mask in the sequence.
        
    Returns:
        list: Modified token_ids with randomly masked tokens.
    """
    length = len(token_ids)
    num_to_mask = int(round(length * mask_ratio))
    if num_to_mask < 1:
        return token_ids
    
    # Select unique random indices to mask
    indices_to_mask = np.random.choice(range(length), num_to_mask, replace=False)
    
    # Use mask_token_id if available, otherwise fallback to unk_token_id
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id
    new_ids = token_ids.copy()
    for idx in indices_to_mask:
        new_ids[idx] = mask_token_id
    return new_ids

def mask_test_dataset(tokenized_dataset, tokenizer, mask_ratio=0.2):
    """
    Applies masking to the 'input_ids' field of the tokenized dataset.
    """
    def mask_function(example):
        new_input_ids = mask_sequence(example["input_ids"], tokenizer, mask_ratio)
        return {"input_ids": new_input_ids}
    
    masked_dataset = tokenized_dataset.map(mask_function)
    return masked_dataset

def evaluate_model_masked(trainer, tokenized_test, test_labels, model_name: str, tokenizer, mask_ratio=0.2):
    """
    Evaluates the model using a masked version of the test set.
    """
    masked_tokenized_test = mask_test_dataset(tokenized_test, tokenizer, mask_ratio)
    test_results = trainer.predict(masked_tokenized_test)
    predictions = np.argmax(test_results.predictions, axis=-1)
    print(len(predictions))
    cm = confusion_matrix(test_labels, predictions)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} with Mask Ratio {mask_ratio}")
    plt.savefig(f"confusion_matrix_masked_{model_name}_{str(int(mask_ratio*10))}.png")
    plt.show()

    print("Test Metrics (Masked Inference):")
    print(f"F1 Score: {test_results.metrics.get('test_f1_score', 'N/A')}")
    print(f"MCC: {matthews_corrcoef(test_labels, predictions)}")

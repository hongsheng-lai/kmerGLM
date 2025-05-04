#!/usr/bin/env python
"""
Script to fine-tune the sketch sequence model using only cuda:0.
This script enforces that only GPU device cuda:0 is used by:
1. Setting the CUDA_VISIBLE_DEVICES environment variable to "0".
2. Explicitly verifying that the device obtained from get_device is cuda:0.
It loads a pre-trained nucleotide transformer model, tokenizes the data using minimizers,
fine-tunes the model on the promoter dataset, evaluates it, and plots the F1 score.
"""
from helper import *
import os
# Restrict CUDA to only the device with ID 0
from transformers import Trainer 

def main():
    # Enforce the use of cuda:2
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set model and training configurations
    model_name = "splice_baseline"
    pretrained_model_name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    num_labels = 3
    batch_size = 64

    # Load the pretrained model and tokenizer
    model, tokenizer = load_saved_model(model_name, device)

    # Load the training and test datasets
    dataset_name = "splice_sites_all"
    test_dataset = load_data(dataset_name, "test")

    # Preprocess datasets using minimizers for both training and testing
    tokenized_test, test_labels = preprocess_test_data(test_dataset, tokenizer)

    # Set the training arguments with your specified batch size
    training_args = get_training_args(model_name, batch_size, num_labels)

    # Fine-tune the model using the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mcc,
        eval_dataset=tokenized_test,
    )

    for mask_ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        # Evaluate the model on the test set with masked inference
        print(f"Evaluating model with mask ratio: {mask_ratio}")
        evaluate_model_masked(trainer, tokenized_test, test_labels, model_name, tokenizer, mask_ratio=mask_ratio)

if __name__ == '__main__':
    main()

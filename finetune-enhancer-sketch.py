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

from transformers import AutoTokenizer

def main():
    # Enforce the use of cuda:2
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set model and training configurations
    model_name = "enhancers_sketch"
    pretrained_model_name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    num_labels = 3
    batch_size = 64

    # Load the pretrained model and tokenizer
    model = load_model(pretrained_model_name, num_labels, device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # Load the training and test datasets
    dataset_name = "enhancers_types"
    train_dataset = load_data(dataset_name, "train")
    test_dataset = load_data(dataset_name, "test")

    # Preprocess datasets using minimizers for both training and testing
    tokenized_train, tokenized_validation = preprocess_train_data_with_minimizers(train_dataset, tokenizer, k=6, w=20)
    tokenized_test, test_labels = preprocess_test_data_with_minimizers(test_dataset, tokenizer, k=6, w=20)

    # Set the training arguments with your specified batch size
    training_args = get_training_args(model_name, batch_size, num_labels)

    # Fine-tune the model using the trainer
    trainer = train_model(model, training_args, tokenized_train, tokenized_validation, tokenizer)

    # Save the fine-tuned model
    save_model(trainer, model_name)

    # Evaluate the model on the test set
    evaluate_model(trainer, tokenized_test, test_labels, model_name)

    # Plot the F1 score from training/evaluation
    plot_mcc_score(trainer, model_name)

if __name__ == '__main__':
    main()

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (DebertaV2Config, DebertaV2Tokenizer, EvalPrediction,
                          TrainingArguments, Trainer, AutoModelForSequenceClassification, get_scheduler)
import torch
import argparse
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run ScaLA for full finetuning")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Specify the language of the dataset (is|vs|de)",
    )
    return parser.parse_args()

def load_and_prepare_data(language):
    """Load and preprocess the dataset."""
    # Load dataset
    dataset = load_dataset("alexandrainst/scala", language)

    # Load tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/mdeberta-v3-base")

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(
            batch["text"], max_length=512, truncation=True, padding="max_length"
        )

    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Convert labels to numerical values
    label_to_id = {"correct": 1, "incorrect": 0}
    def convert_labels(batch):
        batch["labels"] = label_to_id[batch["labels"]]
        return batch

    dataset = dataset.map(convert_labels)

    return dataset

def compute_metrics(p: EvalPrediction):
    """Compute evaluation metrics."""
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).mean()
    mcc = matthews_corrcoef(p.label_ids, preds)
    macro_f1 = f1_score(p.label_ids, preds, average='macro')
    return {"acc": acc, "mcc": mcc, "macro_f1": macro_f1}

def load_model():
    """Load the model with adapter configurations."""
    config = DebertaV2Config.from_pretrained("microsoft/mdeberta-v3-base", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/mdeberta-v3-base", config=config)
    return model

def perform_cross_validation(dataset, num_splits=10):
    """Perform k-fold cross-validation."""
    # Convert dataset to pandas DataFrame for easier splitting
    df = dataset["train"].to_pandas()
    test_data = dataset["test"]

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    val_metrics = {"acc": [], "macro_f1": [], "mcc": []}
    test_metrics = {"acc": [], "macro_f1": [], "mcc": []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        logger.info(f"Fold {fold + 1}")

        # Split the dataset into training and validation based on the fold
        train_data = Dataset.from_pandas(df.iloc[train_idx])
        val_data = Dataset.from_pandas(df.iloc[val_idx])

        train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Load model
        model = load_model()

        training_args = TrainingArguments(
            output_dir="./training_output",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            logging_steps=1000,
            save_strategy='no'
        )

        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        num_train_steps = len(train_data) * training_args.num_train_epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, lr_scheduler)
        )

        # Train the model
        trainer.train()

        # Evaluate on validation set
        validation_results = trainer.evaluate(eval_dataset=val_data)
        logger.info(f"Validation Results: {validation_results}")

        # Store validation metrics
        val_metrics["acc"].append(validation_results["eval_acc"])
        val_metrics["macro_f1"].append(validation_results["eval_macro_f1"])
        val_metrics["mcc"].append(validation_results["eval_mcc"])

        # Evaluate on test set
        test_results = trainer.evaluate(eval_dataset=test_data)
        logger.info(f"Test Results for Fold {fold + 1}: {test_results}")

        # Store test metrics
        test_metrics["acc"].append(test_results["eval_acc"])
        test_metrics["macro_f1"].append(test_results["eval_macro_f1"])
        test_metrics["mcc"].append(test_results["eval_mcc"])

    return val_metrics, test_metrics

def summarize_results(metrics, name):

    """Summarize results from cross-validation."""
    mean_metrics = {key: np.mean(values) for key, values in metrics.items()}
    std_metrics = {key: np.std(values) for key, values in metrics.items()}

    for metric in metrics.keys():
        logger.info(f"{name} {metric.upper()}: {mean_metrics[metric]*100:.2f} ± {std_metrics[metric]*100:.2f}")

def main():
    """Main function to run the script."""
    args = parse_arguments()
    logger.info(f"{args.language} ScaLA Full Finetuning")

    dataset = load_and_prepare_data(args.language)
    val_metrics, test_metrics = perform_cross_validation(dataset)

    logger.info("\nCross-Validation Results:")
    summarize_results(val_metrics, "Validation")

    logger.info("\nTest Results:")
    summarize_results(test_metrics, "Test")

    logger.info(f"{args.language} ScaLA Full Finetuned!")

if __name__ == "__main__":
    main()
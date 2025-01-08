import torch
import argparse
import logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, f1_score
from datasets import load_dataset, Dataset
from transformers import (
    DebertaV2Config,
    DebertaV2Tokenizer,
    EvalPrediction,
    TrainingArguments,
    get_scheduler,
)
from adapters import AdapterTrainer, AutoAdapterModel, init

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run ScaLA with specified adapters")
    parser.add_argument(
        "--language_adapter_type",
        type=str,
        required=False,
        help="Specify the type of language adapter (e.g., islandic_lora8)",
    )
    parser.add_argument(
        "--task_adapter_type",
        type=str,
        required=False,
        help="Specify the type of task adapter (e.g., lora)",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Specify the language of the dataset (is|vs|de)",
    )
    return parser.parse_args()


def load_and_prepare_data(language, tokenizer):
    """Load and preprocess the dataset."""
    logger.info("Loading dataset...")
    dataset = load_dataset("alexandrainst/scala", language)

    # Tokenize dataset
    def encode_batch(batch):
        return tokenizer(
            batch["text"], max_length=512, truncation=True, padding="max_length"
        )

    logger.info("Tokenizing dataset...")
    dataset = dataset.map(encode_batch, batched=True)
    dataset = dataset.rename_column("label", "labels")

    # Convert labels to numerical format
    label_to_id = {"correct": 1, "incorrect": 0}

    def convert_labels(batch):
        batch["labels"] = label_to_id[batch["labels"]]
        return batch

    dataset = dataset.map(convert_labels)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    logger.info("Data preparation complete.")
    return dataset


def compute_metrics(p: EvalPrediction):
    """Compute evaluation metrics."""
    preds = np.argmax(p.predictions, axis=1)
    acc = (preds == p.label_ids).mean()
    mcc = matthews_corrcoef(p.label_ids, preds)
    macro_f1 = f1_score(p.label_ids, preds, average="macro")
    return {"acc": acc, "mcc": mcc, "macro_f1": macro_f1}


def perform_cross_validation(dataset, args):
    """Perform k-fold cross-validation."""
    df = dataset["train"].to_pandas()
    test_data = dataset["test"]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    val_metrics = {"acc": [], "macro_f1": [], "mcc": []}
    test_metrics = {"acc": [], "macro_f1": [], "mcc": []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        logger.info(f"Starting Fold {fold + 1}...")
        train_data = Dataset.from_pandas(df.iloc[train_idx])
        val_data = Dataset.from_pandas(df.iloc[val_idx])

        train_data.set_format(type="torch")
        val_data.set_format(type="torch")
        test_data.set_format(type="torch")

        model = load_model_with_adapters(args)

        training_args = TrainingArguments(
        output_dir="./training_output",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        logging_steps = 1000,
        save_strategy='no'
    )

        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
        num_train_steps = len(train_data) * training_args.num_train_epochs

        num_warmup_steps = 0 

        lr_scheduler = get_scheduler(
            "linear",  # Or another scheduler type
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, lr_scheduler),
        )

        trainer.train()
        # Evaluate on validation set
        validation_results = trainer.evaluate(eval_dataset=val_data)
        print(f"Validation Results: {validation_results}")

        # Store validation metrics
        val_metrics["acc"].append(validation_results["eval_acc"])
        val_metrics["macro_f1"].append(validation_results["eval_macro_f1"])
        val_metrics["mcc"].append(validation_results["eval_mcc"])

        # Evaluate on test set
        test_results = trainer.evaluate(eval_dataset=test_data)
        print(f"Test Results for Fold {fold + 1}: {test_results}")

        # Store test metrics
        test_metrics["acc"].append(test_results["eval_acc"])
        test_metrics["macro_f1"].append(test_results["eval_macro_f1"])
        test_metrics["mcc"].append(test_results["eval_mcc"])

    return val_metrics, test_metrics



def load_model_with_adapters(args):
    """Load model and adapters."""
    model_name = "microsoft/mdeberta-v3-base"
    config = DebertaV2Config.from_pretrained(model_name, num_labels=2)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)

    # Initialize adapters
    init(model)
    model.load_adapter(f"rominaoji/{args.language_adapter_type}")
    model.add_classification_head(
        "taskadapter", num_labels=2, id2label={0: "incorrect", 1: "correct"}
    )
    model.add_adapter("taskadapter", config=args.task_adapter_type)
    model.set_active_adapters(["taskadapter"])
    model.train_adapter("taskadapter")
    return model


def summarize_results(metrics):
    """Summarize results from cross-validation."""
    mean_metrics = {key: np.mean(values) for key, values in metrics.items()}
    std_metrics = {key: np.std(values) for key, values in metrics.items()}

    for metric in metrics.keys():
        print(f"{metric.upper()}: {mean_metrics[metric]*100:.2f} ± {std_metrics[metric]*100:.2f}")


def main():
    args = parse_arguments()
    logger.info(
        f"Task Adapter: {args.task_adapter_type}, "
        f"Language Adapter: {args.language_adapter_type}"
        f"Language: {args.language}"
    )

    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    dataset = load_and_prepare_data(args.language, tokenizer)

    val_results, test_results = perform_cross_validation(dataset, args)
    logger.info("Validation Results Summary:")
    summarize_results(val_results)

    logger.info("Test Results Summary:")
    summarize_results(test_results)


if __name__ == "__main__":
    main()

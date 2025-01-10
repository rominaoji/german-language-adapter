from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
import adapters
from adapters import AdapterTrainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, get_scheduler
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_metric
from sklearn.model_selection import KFold
from tqdm import tqdm
import argparse
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run QA with specified adapters")
    parser.add_argument('--language_adapter_type', type=str, required=False,
                        help='Specify the type of language adapter (e.g., swedish_lora8)')
    parser.add_argument('--task_adapter_type', type=str, required=True,
                        help='Specify the type of task adapter (e.g., lora)')
    return parser.parse_args()


def load_and_prepare_dataset():
    dataset = load_dataset('KBLab/sucx3_ner', 'original_cased')

    unique_labels = set()
    for example in dataset["train"]:
        unique_labels.update(example["ner_tags"])
    label_names = sorted(unique_labels)
    id_2_label = {id_: label for id_, label in enumerate(label_names)}
    label_2_id = {label: id_ for id_, label in enumerate(label_names)}
    return dataset, label_names, id_2_label, label_2_id



def tokenize_adjust_labels(all_samples_per_split, tokenizer, label_2_id):
    tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True, padding=True, truncation=True)
    total_adjusted_labels = []

    for k in range(len(tokenized_samples["input_ids"])):
        word_ids = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = [label_2_id[label] for label in all_samples_per_split["ner_tags"][k]]
        adjusted_label_ids = []
        prev_wid = -1

        for wid in word_ids:
            if wid is None:
                adjusted_label_ids.append(-100)
            elif wid != prev_wid:
                adjusted_label_ids.append(existing_label_ids[wid])
                prev_wid = wid
            else:
                adjusted_label_ids.append(existing_label_ids[wid])

        total_adjusted_labels.append(adjusted_label_ids)

    tokenized_samples["labels"] = total_adjusted_labels
    return tokenized_samples



def compute_metrics(p, label_names, metric):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    return flattened_results

def train_and_evaluate_kfold(args, dataset, tokenizer, label_names, id_2_label, label_2_id):
    tokenized_dataset = dataset.map(lambda x: tokenize_adjust_labels(x, tokenizer, label_2_id), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_val_metrics, all_test_metrics = [], []

    for fold, (train_index, val_index) in enumerate(kf.split(tokenized_dataset["train"])):
        logger.info(f"Fold {fold + 1}")
        train_fold = tokenized_dataset["train"].select(train_index)
        val_fold = tokenized_dataset["train"].select(val_index)

        config = AutoConfig.from_pretrained(model_name, num_labels=len(label_names), label2id=label_2_id, id2label=id_2_label)
        model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

        adapters.init(model)
        if args.language_adapter_type:
            model.load_adapter("rominaoji/" + args.language_adapter_type)
        model.add_adapter("ner", config=args.task_adapter_type)
        model.train_adapter("ner")

        training_args = TrainingArguments(
            output_dir="./training_output",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            logging_steps=1000,
            save_strategy='no'
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        num_train_steps = len(train_fold) * training_args.num_train_epochs
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, label_names, metric),
            optimizers=(optimizer, lr_scheduler)
        )

        trainer.train()
        all_val_metrics.append(trainer.evaluate(eval_dataset=tokenized_dataset["validation"]))
        logger.info(f"Validation Metrics: {all_val_metrics[-1]}")
        all_test_metrics.append(trainer.evaluate(eval_dataset=tokenized_dataset["test"]))
        logger.info(f"Test Metrics: {all_test_metrics[-1]}")

    return all_val_metrics, all_test_metrics


def calculate_mean_std(metrics, key):
    mean = np.mean([metric[key] for metric in metrics])
    std = np.std([metric[key] for metric in metrics])
    return mean, std


def main():
    args = parse_arguments()

    logger.info(f"German NER task adapter: {args.task_adapter_type}, language adapter: {args.language_adapter_type}")

    dataset, label_names, id_2_label, label_2_id = load_and_prepare_dataset()

    global model_name
    model_name = "microsoft/mdeberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_val_metrics, all_test_metrics = train_and_evaluate_kfold(args, dataset, tokenizer, label_names, id_2_label, label_2_id)
    mean_val_f1, std_val_f1 = calculate_mean_std(all_val_metrics, "eval_overall_f1")

    logger.info(f"K-Fold Validation F1: {mean_val_f1*100:.2f} ± {std_val_f1*100:.2f}")

    # Test metrics
    mean_test_f1, std_test_f1 = calculate_mean_std(all_test_metrics, "eval_overall_f1")

    logger.info(f"K-Fold Test F1: {mean_test_f1*100:.2f} ± {std_test_f1*100:.2f}")
    
    logger.info(f"Swedish NER task adapter: {args.task_adapter_type}, language adapter: {args.language_adapter_type}")

if __name__ == "__main__":
    main()
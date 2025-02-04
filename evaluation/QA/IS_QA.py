import collections
import torch
import adapters
import argparse
import evaluate
import numpy as np
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from adapters import AdapterTrainer
from datasets import load_dataset
from transformers import (AutoConfig, AutoTokenizer,
                          AutoModelForQuestionAnswering,
                          TrainingArguments,
                          get_scheduler)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run QA with specified adapters")
    parser.add_argument('--language_adapter_type', type=str, required=False,
                        help='Specify the type of language adapter (e.g., icelandic_lora8)')
    parser.add_argument('--task_adapter_type', type=str, required=False,
                        help='Specify the type of task adapter (e.g., lora)')
    return parser.parse_args()

def load_and_prepare_dataset():
    raw_datasets = load_dataset("vesteinn/icelandic-qa-NQiI")
    raw_datasets = raw_datasets.filter(lambda x: len(x["answers"]["text"]) > 0)
    return raw_datasets

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        if len(answer["answer_start"]) == 0 or len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    theoretical_answers = []

    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        ground_truths = example["answers"]["text"]

        if not ground_truths:
            continue

        answers = []
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        if answers:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
            theoretical_answers.append(
                {"id": example_id, "answers": example["answers"]}
            )

    if predicted_answers and theoretical_answers:
        return metric.compute(
            predictions=predicted_answers, references=theoretical_answers
        )
    else:
        return {"error": "No valid predictions or theoretical answers"}

def train_and_evaluate_kfold(args, train_dataset, validation_dataset, test_dataset, raw_datasets):
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_val_metrics = []
    all_test_metrics = []

    for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}")
        train_fold = train_dataset.select(train_index)
        val_fold = train_dataset.select(val_index)

        config1 = AutoConfig.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config1)
        adapters.init(model)

        if args.language_adapter_type != None:
            model.load_adapter("rominaoji/" + args.language_adapter_type)

        model.add_adapter("qa", config=args.task_adapter_type)
        model.train_adapter("qa")

        training_args = TrainingArguments(
            output_dir="./training_output",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            logging_steps = 1000,
            save_strategy='no'
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        num_train_steps = len(train_fold) * training_args.num_train_epochs
        print(f"Total number of training steps: {num_train_steps}")
        num_warmup_steps = 0
        print(f"Total number of warmup steps: {num_warmup_steps}")

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )       

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, lr_scheduler)
        )

        print(trainer.train())

        val_predictions = trainer.predict(validation_dataset)
        val_start_logits, val_end_logits = val_predictions.predictions
        val_metrics = compute_metrics(
            val_start_logits, val_end_logits, validation_dataset, raw_datasets["validation"]
        )
        all_val_metrics.append(val_metrics)
        print("Validation Metrics:", val_metrics)

        test_predictions = trainer.predict(test_dataset)
        test_start_logits, test_end_logits = test_predictions.predictions
        test_metrics = compute_metrics(
            test_start_logits, test_end_logits, test_dataset, raw_datasets["test"]
        )
        all_test_metrics.append(test_metrics)
        print("Test Metrics:", test_metrics)

    return all_val_metrics, all_test_metrics

def calculate_mean_std(metrics):
    mean_exact_match = np.mean([metric["exact_match"] for metric in metrics])
    std_exact_match = np.std([metric["exact_match"] for metric in metrics])
    mean_f1 = np.mean([metric["f1"] for metric in metrics])
    std_f1 = np.std([metric["f1"] for metric in metrics])

    return mean_exact_match, std_exact_match, mean_f1, std_f1

def main():
    args = parse_arguments()

    print(f"Icelandic QA task adapter: {args.task_adapter_type} language adapter: {args.language_adapter_type}")

    raw_datasets = load_and_prepare_dataset()

    global tokenizer, max_length, stride, n_best, max_answer_length, metric, model_name
    model_name = "microsoft/mdeberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 384
    stride = 128
    n_best = 20
    max_answer_length = 30
    metric = evaluate.load("squad")
    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    test_dataset = raw_datasets["test"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
    )

    all_val_metrics, all_test_metrics = train_and_evaluate_kfold(args, train_dataset, validation_dataset, test_dataset, raw_datasets)

    mean_val_exact_match, std_val_exact_match, mean_val_f1, std_val_f1 = calculate_mean_std(all_val_metrics)
    print(f"K-Fold Validation Exact Match: {mean_val_exact_match:.2f} ± {std_val_exact_match:.2f}")
    print(f"K-Fold Validation F1 Score: {mean_val_f1:.2f} ± {std_val_f1:.2f}")

    mean_test_exact_match, std_test_exact_match, mean_test_f1, std_test_f1 = calculate_mean_std(all_test_metrics)
    print(f"K-Fold Test Exact Match: {mean_test_exact_match:.2f} ± {std_test_exact_match:.2f}")
    print(f"K-Fold Test F1 Score: {mean_test_f1:.2f} ± {std_test_f1:.2f}")

    print(f"Icelandic QA task adapter: {args.task_adapter_type} language adapter: {args.language_adapter_type}")

if __name__ == "__main__":
    main()

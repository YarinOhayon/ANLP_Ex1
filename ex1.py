import torch
import numpy as np
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score
from pathlib import Path
import wandb

# ------------------------ Parse the Argument ------------------------ #
# -1 means "use all"
parser = argparse.ArgumentParser()
parser.add_argument("--max_train_samples", type=int, default=-1)
parser.add_argument("--max_eval_samples", type=int, default=-1)
parser.add_argument("--max_predict_samples", type=int, default=-1)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--model_path", type=str, default="")
args = parser.parse_args()


# # Simulate command line arguments within notebook
# # You can modify these arguments as needed
# args_str = "--do_train --do_predict --num_train_epochs 3 --lr 2e-5 --batch_size 16 --max_train_samples 1000 --max_eval_samples 200 --max_predict_samples 100"
# args = parser.parse_args(args_str.split())


# ------------------------ Setup ------------------------ #
# naming the run as 'run_#epochs_learningRate_batchSize'
current_run_name = f"run_{args.num_train_epochs}_lr{args.lr}_bs{args.batch_size}"
wandb.init(project="ANLP-Ex1", name=current_run_name)

# Load the MRPC dataset
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ------------------------ Tokenization ------------------------ #
def tokenize_fn(example):
    # Tokenize with truncation only (no padding here; handled later)
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ------------------------ Sample Subsetting ------------------------ #
# If max_train_samples, max_eval_samples, or max_predict_samples are set to -1, we use the full dataset, otherwise we subset the dataset
if args.max_train_samples != -1:
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(args.max_train_samples))
if args.max_eval_samples != -1:
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(args.max_eval_samples))
if args.max_predict_samples != -1:
    raw_datasets["test"] = raw_datasets["test"].select(range(args.max_predict_samples))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # for the dynamic padding

# ------------------------ Model ------------------------ #
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ------------------------ Training Arguments ------------------------ #
training_args = TrainingArguments(
    output_dir="./results/" + current_run_name,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="wandb",
    save_total_limit=1
)

def compute_metrics(eval_pred):
    # Compute accuracy for the evaluation
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ------------------------ Training ------------------------ #
if args.do_train:
    trainer.train()
    trainer.save_model(training_args.output_dir)
    evaluation_data = trainer.evaluate()
    print(f"Validation accuracy: {evaluation_data['eval_accuracy']:.4f}")

# ------------------------ Prediction ------------------------ #
if args.do_predict:
    # Load the model from the path that was given
    if args.model_path:
        checkpoint_paths = [Path(args.model_path)]
    else:
        checkpoint_paths = list(Path("./results").glob("*/"))
    results = {}
    for ckpt_path in checkpoint_paths:
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Predicts labels for each test example with that checkpoint
        preds = []
        for example in raw_datasets["test"]:
            inputs = tokenizer(example["sentence1"], example["sentence2"], truncation=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
            preds.append(pred)

        test_labels = raw_datasets["test"]["label"]
        acc = accuracy_score(test_labels, preds)
        results[ckpt_path.name] = {"accuracy": acc, "preds": preds}

    # Sort the results by accuracy and select the best one
    sorted_runs = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    best_name, best = sorted_runs[0]
    best_preds = best["preds"]

    with open("predictions.txt", "w") as f:
        for i, example in enumerate(raw_datasets["test"]):
            pred_label = best_preds[i]
            line = f"{example['sentence1']}###{example['sentence2']}###{pred_label}\n"
            f.write(line)
    # print("you were supposed to write the predictions to a file")
# Close the wandb run
wandb.finish()
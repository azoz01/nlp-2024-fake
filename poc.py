from typing import Mapping

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

COLUMNS = [
    "statement_id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "true_cnt",
    "false_cnt",
    "half_true_cnt",
    "mostly_true_cnt",
    "pants_on_fire_cnt",
    "context",
]
LABEL_MAPPING = {
    "false": 1,
    "pants-fire": 1,
    "mostly-true": 0,
    "true": 0,
}
STATEMENT_COLUMN = "statement"
LABEL_COLUMN = "label"
MODEL_ID = "roberta-base"


def read_data(
    path: str,
    columns: list[str] = COLUMNS,
    label_mapping: Mapping[str, int] = LABEL_MAPPING,
) -> pd.DataFrame:
    df = pd.read_table(path, header=None, names=columns)
    df = df[[STATEMENT_COLUMN, LABEL_COLUMN]]
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(label_mapping)
    df = df.loc[~df[LABEL_COLUMN].isnull()]
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    assert (df.isna().sum() == 0).all()
    assert (
        df[LABEL_COLUMN].drop_duplicates().sort_values().values == [0, 1]
    ).all()
    df = df.rename(columns={STATEMENT_COLUMN: "text"})
    return df


logger.info("Loading data")
train = read_data("data/split_raw/train.tsv")
valid = read_data("data/split_raw/valid.tsv")
test = read_data("data/split_raw/test.tsv")

logger.info("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def prepare_data_for_fine_tuning(df: pd.DataFrame):
    dataset = Dataset.from_pandas(df)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            max_length=256,
            truncation=True,
        )

    dataset = dataset.map(tokenize)
    dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )
    return dataset


logger.info("Preparing data for fine-tuning")
train_dataset = prepare_data_for_fine_tuning(train)
valid_dataset = prepare_data_for_fine_tuning(valid)
test_dataset = prepare_data_for_fine_tuning(test)


logger.info("Preparing model for fine-tuning")
metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


config = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, config=config
)
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="tensorboard",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)
logger.info("Training model")
trainer.train()

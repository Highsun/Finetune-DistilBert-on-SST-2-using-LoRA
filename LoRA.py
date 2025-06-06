import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

# Load dataset (SST-2)
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True)

dataset = dataset.map(preprocess_function, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Load base model and apply LoRA
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    # modules_to_save=[] # Only save LoRA layers
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Define evaluation metric
metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="no",
    save_strategy="no",
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print("Eval results:", eval_results)

# Save model
if not os.path.exists("./model"):
    os.makedirs("./model")

save_path = os.path.join("./model", "lora_distilbert_sst2")
trainer.save_model(save_path)
print(f"[\u2713] Training complete. Model saved to {save_path}")
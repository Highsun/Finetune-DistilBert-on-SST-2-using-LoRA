import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    default_data_collator,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# LoRA Linear Layer
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        if r > 0:
            self.A = nn.Parameter(torch.empty(r, in_features))
            self.B = nn.Parameter(torch.empty(out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.A = None
            self.B = None

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora = self.dropout(x) @ self.A.T @ self.B.T
            return base + self.alpha * lora
        return base

# Replace Linear Layers with LoRA
def replace_linear_with_lora(model, r=8, alpha=16, dropout=0.1, target_modules=["q_lin", "v_lin"]):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            lora = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=r, alpha=alpha, dropout=dropout
            )
            lora.weight.data = module.weight.data.clone()
            lora.bias.data = module.bias.data.clone()

            parent = model
            for attr in name.split('.')[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, name.split('.')[-1], lora)
    return model

# Load dataset and tokenizer
dataset = load_dataset("glue", "sst2")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def preprocess_function(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], shuffle=True, batch_size=16, collate_fn=default_data_collator)
eval_loader = DataLoader(dataset["validation"], batch_size=16, collate_fn=default_data_collator)

# Load model and apply LoRA
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = replace_linear_with_lora(model, r=8, alpha=16, dropout=0.1, target_modules=["q_lin", "v_lin"])

# Freeze base model
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if any(x in name for x in ['A', 'B']):
        param.requires_grad = True

# Print trainable parameter ratio
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable} / {total} ({trainable / total * 100:.4f}%)")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.01)
model.train()
for epoch in range(3):
    print(f"\n[Epoch {epoch+1}] Training...")
    total_loss = 0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f"Step {step+1} | Loss: {loss.item():.4f}")
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}\n")

# Evaluation
model.eval()
all_preds, all_labels = [], []
for batch in eval_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    labels = batch.get("label", batch.get("labels")).cpu().tolist()
    all_preds.extend(preds)
    all_labels.extend(labels)

acc = accuracy_score(all_labels, all_preds)
print(f"\nValidation Accuracy: {acc * 100:.2f}%")

# Save
if not os.path.exists("./model"):
    os.makedirs("./model")

torch.save(model.state_dict(), "./model/lora_replay_distilbert_sst2.pt")
tokenizer.save_pretrained("./model/lora_replay_distilbert_sst2")
print("[\u2713] Model weights and tokenizer saved.")
import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

import math
import torch
import torch.nn as nn
from torch import device
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast
)
from peft import PeftConfig, PeftModel

def LoRA_eval(sentence, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained("./model/distilbert-base-uncased")

    peft_config = PeftConfig.from_pretrained("./model/lora_distilbert_sst2")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path, num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, "./model/lora_distilbert_sst2")
    model.to(device)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0: "Negative", 1: "Positive"}
    print(f"LoRA Pred label: {label_map[prediction]}\n")

def lora_replay_sentence(sentence, model_dir="model/lora_replay_distilbert_sst2", device="cpu"):
    class LoRALinear(nn.Module):
        def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
            super().__init__()
            self.r = r
            self.alpha = alpha / r
            self.dropout = nn.Dropout(dropout)
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.A = nn.Parameter(torch.empty(r, in_features))
            self.B = nn.Parameter(torch.empty(out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)

        def forward(self, x):
            base = nn.functional.linear(x, self.weight, self.bias)
            lora = self.dropout(x) @ self.A.T @ self.B.T
            return base + self.alpha * lora

    def replace_linear_with_lora(model, r=8, alpha=16, dropout=0.1):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'classifier' not in name:
                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r, alpha=alpha, dropout=dropout
                )
                lora_layer.weight.data = module.weight.data.clone()
                lora_layer.bias.data = module.bias.data.clone()
                parent = model
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split('.')[-1], lora_layer)
        return model

    def load_lora_model(model_dir, device="cpu"):
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        model = replace_linear_with_lora(model, r=8, alpha=16, dropout=0.1)
        weights_path = f"{model_dir}.pt"
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = load_lora_model(model_dir, device)
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=-1).item()

    label_map = {0: "Negative", 1: "Positive"}
    print(f"LoRA_replay Pred label: {label_map[pred_id]}\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while True:
        print("\nPlease input a sentence for sentiment classification (type 'exit' to quit):")
        sentence = input(">> ")
        if sentence.strip().lower() == "exit":
            break
        print("")
        LoRA_eval(sentence, device=device)
        lora_replay_sentence(sentence, device=device)
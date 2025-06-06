# Finetune DistilBERT on SST-2 using LoRA (from scratch)

This project implements parameter-efficient fine-tuning of `distilbert-base-uncased` on the [GLUE/SST-2](https://huggingface.co/datasets/glue/viewer/sst2) dataset using **LoRA (Low-Rank Adaptation)** manually implemented in PyTorch, without using PEFT libraries.

A PEFT-based finetuning script is also provided in `./LoRA.py`; it is more efficient and less resource-consuming, making it suitable for deployment.

## Project Structure

```
.
├── dataset
│   └── glue_sst2_local
├── Google Colab
│   ├── LoRA_replay.ipynb
│   └── LoRA.ipynb
├── LICENSE
├── LoRA_replay.py
├── LoRA.py
├── model
│   ├── distilbert-base-uncased
│   ├── lora_distilbert_sst2
│   ├── lora_replay_distilbert_sst2
│   └── lora_replay_distilbert_sst2.pt
├── Playground.py
├── README.md
└── setup.sh
```

## Features

- Lightweight fine-tuning using manually implemented LoRA layers
- Injects LoRA only into `q_lin` and `v_lin` of the attention blocks
- Prints trainable parameter stats for verification
- Supports inference on custom input sentences

## Dependencies

Using the `setup.sh` to install required packages (works in local or Google Colab):

```bash
chmod +x setup.sh
./setup.sh
```

## Training

### Local

Run the main script to start fine-tuning:

```bash
python LoRA.py # peft version
python LoRA_replay.py # sell implementation version
```

It will:

- Inject LoRA layers into attention heads
- Print trainable parameter ratio
- Train the model on SST-2 for 3 epochs
- Save weights to: `./model/lora_replay_distilbert_sst2.pt`

> **Note:** The contents of the `dataset` and `model` folders can be obtained via [Baidu NetDisk link](url). Meanwhile, you will need to modify the file paths in the code to use the locally imported dataset and pretrained model.
> **Changelog:** The dataset and model files have not been uploaded yet. Please temporarily use the default download method from the Hugging Face Hub. (Just use the orginal code)

### Google Colab

Upload the notebooks to Google Colab and run them using a T4 GPU or TPU instance.

## Inference

Run `Playground.py` to make predictions on your own sentences (only English):

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentence = "The movie was absolutely fantastic!"

LoRA_eval(sentence, device=device)
lora_replay_sentence(sentence, device=device)
```

Expected output:

```
LoRA Pred label: Positive
LoRA_replay Pred label: Positive
```

## LoRA Configuration

LoRA is injected into:

- `distilbert.transformer.layer.*.attention.q_lin`
- `distilbert.transformer.layer.*.attention.v_lin`

Configuration:

- `r = 8`
- `alpha = 16`
- `dropout = 0.1`
- `learning_rate = 2e-4`
- `epoches = 3`

Total trainable parameters ~**147k**, which is just **0.22%** of the full model (~67M), while still achieving **88.76%** accuracy.

## Model Saving

Saved artifacts:

- `lora_replay_distilbert_sst2.pt`: Trained LoRA-injected weights
- Tokenizer files: `tokenizer.json`, `vocab.txt`, etc.

## References

- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [GLUE Benchmark](https://gluebenchmark.com/tasks)

## Contact

Feel free to open issues or pull requests in [this repository](https://github.com/Highsun/Finetune-DistilBert-on-SST-2-using-LoRA) for suggestions or questions.

Cooperations please email at highsun910@gmail.com

# Module 8: NLP & Transformers

## Overview

Raikura provides built-in NLP support with deep integration for HuggingFace Transformers. It enables classification and sequence tasks using pretrained models like BERT, RoBERTa, or DistilBERT, as well as compatibility with Raikura's pipelines.

---

## Capabilities

* Pretrained transformer support via `transformers` (e.g. BERT, RoBERTa)
* Tokenization, padding, truncation
* Fine-tuning via Trainer API
* Text preprocessing, feature injection
* Multiclass and binary classification

---

## TransformerClassifier API

**File**: `models/transformer.py`

```python
class TransformerClassifier:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def fit(self, texts, labels):
        dataset = build_dataset(texts, labels, self.tokenizer)
        trainer = Trainer(model=self.model, args=TrainingArguments(...), train_dataset=dataset)
        trainer.train()

    def predict(self, texts):
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**tokens)
        return torch.argmax(outputs.logits, axis=1).numpy()
```

---

## Example Usage

```python
from raikura.models.transformer import TransformerClassifier
model = TransformerClassifier(model_name="bert-base-uncased")
model.fit(train_texts, train_labels)
preds = model.predict(test_texts)
```

---

## Tokenization Pipeline

Raikura can extract token embeddings from a text column and inject them into a tabular pipeline via:

```python
text_embeddings = tokenize_text(X["text_col"], model_name="distilbert-base-uncased")
X = pd.concat([X.drop("text_col", axis=1), text_embeddings], axis=1)
```

---

## Configuration Example

```yaml
pipeline:
  model_type: bert_classifier
  preprocessing:
    text_column: "description"
    text_model: "bert-base-uncased"
```

---

## CLI Example

```bash
raikura nlp --config config_nlp.yaml
```

---

## Advanced Features

* Works with any HuggingFace checkpoint
* Supports transfer learning and freezing layers
* Can be embedded in ensemble/multimodal pipelines

---

## Summary

Raikura makes it seamless to go from raw text to state-of-the-art transformer-based classification. With minimal setup, users can fine-tune BERT models or inject embeddings into hybrid pipelines for powerful NLP tasks.

Next Module: **Multimodal Fusion (Text + Tabular)**

---
title: "Train BERT to classify 22 languages"
date: 2022-01-14
toc: true
toc_label: "Contents"
toc_sticky: True
published: true
excerpt: "Train a transformer (BERT) from HuggingFace Transformers to classify 22 different languages. Our model achieves an accuracy of 96% on the test set."
categories:
  - Programming
  - Machine Learning
tags:
  - Python
  - PyTorch
  - NLP
  - Transformers
---

Transformers are a relatively new type of models (at the time of writing), which have overtaken the NLP world as the best model. These models are much more efficient than RNN/LSTM's and offer better performance on datasets.

In this article, we will be training a transformer (BERT) on a dataset of 22 languages.

The dataset can be found [here](https://www.kaggle.com/zarajamshaid/language-identification-datasst). More information about Bert can be found [here](https://arxiv.org/abs/1810.04805).

## Importing Libraries

We will need the following libraries for this project

- `pandas` - `pip install pandas` (Pandas)
- `torch` - `pip install torch` (PyTorch
- `datasets` - `pip install datasets` (A library to load NLP datasets)
- `transformers` - `pip install transformers` (Huggingface Transformers)
- Scikit-learn

```python
import pandas as pd # data processing, CSV file I/O
import datasets # pip install datasets
import os
```

The following block imports the main libraries, the ones that define the model, process the data, and train it.
```python
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```

## Loading in the data

```python
# Replace the path with wherever your dataset is
# This is for a Kaggle Kernel
df = pd.read_csv("../input/language-identification-datasst/dataset.csv")
```

### Preprocess Labels

```python
label_encoder = preprocessing.LabelEncoder()
encoded_labels = label_encoder.fit_transform(df["language"])
df["language"] = encoded_labels
```

```python
languages = label_encoder.classes_
print(languages)
```

```
[OUTPUT]
['Arabic' 'Chinese' 'Dutch' 'English' 'Estonian' 'French' 'Hindi'
 'Indonesian' 'Japanese' 'Korean' 'Latin' 'Persian' 'Portugese' 'Pushto'
```

### Converting it into a Transformers readable format

```python
from datasets import Dataset, DatasetDict
raw_dataset = Dataset.from_pandas(df)
raw_datasets = raw_dataset.train_test_split(test_size=0.12)
# We are going to tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

Now, we are going to tokenize the dataset, so we can feed it into model. 

```python
def tokenize_function(examples):
    return tokenizer(examples["Text"], padding="max_length", truncation=True)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
```

After that, the labels need to be processed

```python
def label_encoding(examples):
    return tokenizer(examples["Text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

## Model

The following cell loads the model `bert-base-cased`
```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=len(languages))
```

Convert the dataset into PyTorch format

```python
tokenized_datasets = tokenized_datasets.remove_columns(["Text"])
tokenized_datasets = tokenized_datasets.rename_column("language", "labels")
tokenized_datasets.set_format("torch")
```

### Create smaller training dataset, optional

To speed up training, we can create a smaller training dataset, since this is just for education purposes.
```python
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(12000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]
```

### Optimimizer

We use `AdamW`, a variant of Adam with weight decay.

```python
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)
```

> Note how we train the entire model, according to HuggingFace, it is better to tune the entire transformer rathe than the last layer like in CNN transfer learning.

## Training

Now, its time for the training portion of this tutorial. We start by defining data loaders. 

```python
from torch.utils.data import DataLoader
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(full_eval_dataset, batch_size=8)
```

> Replace `small_train_dataset` with `full_train_dataset` if you wish to train on the full dataset

### Hyperparameters and LR Scheduler

```python
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
```

We need to only train for 3 epochs, and we initialize a learning rate scheduler using the `get_scheduler` function imported from `transformers`

If you have a GPU, move the model to the GPU, because it will make training way faster. You can get a free GPU from Kaggle or Google Colab, or pay for a GPU on a platform like AWS or GCP.

```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
```

### Training loop

Writing the PyTorch training loop is very simple. It is a barebones training loop, with just the basic operations.

```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

For each epoch, we iterate over the data loader and create a batch that is easy for the model to use. We then pass the data into the model and take the loss. The model calculates the loss for us automatically, we don't have to define our own loss function. We then calculate the gradients and update the weights. 

I added a progress bar from `tqdm` so that we can see how training is progressing, it also gives an estimate as to when training will be done. 

## Evaluation

After the model is trained, we can run it on the evaluation dataset. The `datasets` library provides an easy way to calculate the accuracy of evaluation dataset.

```python
from datasets import load_metric
metric= load_metric("accuracy")
model.eval()
for batch in tqdm(eval_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

```
[OUTPUT]
{'accuracy': 0.9696969696969697}
```

Our model gets an accuracy of $96$%, which is very good. We can now proceed to test it on a few sentences

### Trying the model out

First, move the model to the CPU, its not necessary however.
```python
model = model.cpu()
```

We define a simple `predict` function that makes it easy to predict the language of a sentence

```python
def predict(sentence):
    tokenized = tokenizer(sentence, return_tensors="pt")
    outputs = model(**tokenized)
    return languages[outputs.logits.argmax(dim=1)]
```

Now, try out the prediction function. 
```python
sentence1 = "in war resolution, in defeat defiance, in victory magnanimity"
predict(sentence1) # English

sentence2 = "en la guerra resolución en la derrota desafío en la victoria magnanimidad" #spanish
predict(sentence2) # Spanish
```

That's it and thank you for reading this tutorial.
# Exercise: Create a BERT sentiment classifier

In this exercise, you will create a BERT sentiment classifier (actually DistilBERT) using the [Hugging Face Transformers](https://huggingface.co/transformers/) library. 

You will use the [IMDB movie review dataset](https://huggingface.co/datasets/imdb) to train and evaluate your model. The IMDB dataset contains movie reviews that are labeled as either positive or negative. 


```python
# Install the required version of datasets in case you have an older version
# You will need to choose "Kernel > Restart Kernel" from the menu after executing this cell
! pip install -q "datasets==2.15.0"
```


```python
!pip install -q --upgrade datasets

```


```python
# Import the datasets and transformers packages
from datasets import load_dataset

# Load the train and test splits of the imdb dataset
splits = ["train", "test"]
ds = {split: ds for split, ds in zip(splits, load_dataset("imdb", split=splits))}

# Thin out the dataset to make it run faster for this example
for split in splits:
    ds[split] = ds[split].shuffle(seed=42).select(range(500))

# Show the dataset
ds
```




    {'train': Dataset({
         features: ['text', 'label'],
         num_rows: 500
     }),
     'test': Dataset({
         features: ['text', 'label'],
         num_rows: 500
     })}



## Pre-process datasets

Now we are going to process our datasets by converting all the text into tokens for our models. You may ask, why isn't the text converted already? Well, different models may use different tokenizers, so by converting at train time we retain more flexibility.


```python
# Replace <MASK> with your code that constructs a query to send to the LLM


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    """Preprocess the imdb dataset by returning tokenized examples."""
    return tokenizer(
        examples["text"],  # Access the "text" field in the dataset
        padding="max_length",  # Pad all sequences to the model's maximum length
        truncation=True,       # Truncate sequences longer than the model's max input length
        max_length=512,        # Set the max length for padding/truncation
    )


tokenized_ds = {}
for split in splits:
    tokenized_ds[split] = ds[split].map(preprocess_function, batched=True)


# Check that we tokenized the examples properly
assert tokenized_ds["train"][0]["input_ids"][:5] == [101, 2045, 2003, 2053, 7189]

# Show the first example of the tokenized training set
print(tokenized_ds["train"][0]["input_ids"])
```

    The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.



    0it [00:00, ?it/s]


    /home/student/.local/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(



    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/483 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



    Map:   0%|          | 0/500 [00:00<?, ? examples/s]



    Map:   0%|          | 0/500 [00:00<?, ? examples/s]


    [101, 2045, 2003, 2053, 7189, 2012, 2035, 2090, 3481, 3771, 1998, 6337, 2099, 2021, 1996, 2755, 2008, 2119, 2024, 2610, 2186, 2055, 6355, 6997, 1012, 6337, 2099, 3504, 15594, 2100, 1010, 3481, 3771, 3504, 4438, 1012, 6337, 2099, 14811, 2024, 3243, 3722, 1012, 3481, 3771, 1005, 1055, 5436, 2024, 2521, 2062, 8552, 1012, 1012, 1012, 3481, 3771, 3504, 2062, 2066, 3539, 8343, 1010, 2065, 2057, 2031, 2000, 3962, 12319, 1012, 1012, 1012, 1996, 2364, 2839, 2003, 5410, 1998, 6881, 2080, 1010, 2021, 2031, 1000, 17936, 6767, 7054, 3401, 1000, 1012, 2111, 2066, 2000, 12826, 1010, 2000, 3648, 1010, 2000, 16157, 1012, 2129, 2055, 2074, 9107, 1029, 6057, 2518, 2205, 1010, 2111, 3015, 3481, 3771, 3504, 2137, 2021, 1010, 2006, 1996, 2060, 2192, 1010, 9177, 2027, 9544, 2137, 2186, 1006, 999, 999, 999, 1007, 1012, 2672, 2009, 1005, 1055, 1996, 2653, 1010, 2030, 1996, 4382, 1010, 2021, 1045, 2228, 2023, 2186, 2003, 2062, 2394, 2084, 2137, 1012, 2011, 1996, 2126, 1010, 1996, 5889, 2024, 2428, 2204, 1998, 6057, 1012, 1996, 3772, 2003, 2025, 23105, 2012, 2035, 1012, 1012, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


## Load and set up the model

We will now load the model and freeze most of the parameters of the model: everything except the classification head.


```python
# Replace <MASK> with your code freezes the base model

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},  # For converting predictions to strings
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

# Freeze all the parameters of the base model
# Hint: Check the documentation at https://huggingface.co/transformers/v4.2.2/training.html
for param in model.base_model.parameters():
    param.requires_grad = False  # Ensure base model parameters are not updated during training

model.classifier
```

    /home/student/.local/lib/python3.10/site-packages/huggingface_hub/file_download.py:795: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(



    model.safetensors:   0%|          | 0.00/268M [00:00<?, ?B/s]


    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.bias', 'classifier.weight', 'pre_classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.





    Linear(in_features=768, out_features=2, bias=True)




```python
print(model)
```

    DistilBertForSequenceClassification(
      (distilbert): DistilBertModel(
        (embeddings): Embeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (transformer): Transformer(
          (layer): ModuleList(
            (0-5): 6 x TransformerBlock(
              (attention): MultiHeadSelfAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (q_lin): Linear(in_features=768, out_features=768, bias=True)
                (k_lin): Linear(in_features=768, out_features=768, bias=True)
                (v_lin): Linear(in_features=768, out_features=768, bias=True)
                (out_lin): Linear(in_features=768, out_features=768, bias=True)
              )
              (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (ffn): FFN(
                (dropout): Dropout(p=0.1, inplace=False)
                (lin1): Linear(in_features=768, out_features=3072, bias=True)
                (lin2): Linear(in_features=3072, out_features=768, bias=True)
                (activation): GELUActivation()
              )
              (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
          )
        )
      )
      (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
      (classifier): Linear(in_features=768, out_features=2, bias=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )


## Let's train it!

Now it's time to train our model. We'll use the `Trainer` class from the 🤗 Transformers library to do this. The `Trainer` class provides a high-level API that abstracts away a lot of the training loop.

First we'll define a function to compute our accuracy metreic then we make the `Trainer`.

Let's take this opportunity to learn about the `DataCollator`. According to the HuggingFace documentation:

> Data collators are objects that will form a batch by using a list of dataset elements as input. These elements are of the same type as the elements of train_dataset or eval_dataset.

> To be able to build batches, data collators may apply some processing (like padding).



```python
# Replace <MASK> with your DataCollatorWithPadding argument(s)

import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
# Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/sentiment_analysis",
        learning_rate=2e-3,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
    compute_metrics=compute_metrics,
)

trainer.train()
```

    You're using a DistilBertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.




    <div>

      <progress value='125' max='125' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [125/125 00:16, Epoch 1/1]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.476473</td>
      <td>0.792000</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=125, training_loss=0.5901978759765625, metrics={'train_runtime': 17.6104, 'train_samples_per_second': 28.392, 'train_steps_per_second': 7.098, 'total_flos': 66233699328000.0, 'train_loss': 0.5901978759765625, 'epoch': 1.0})



## Evaluate the model

Evaluating the model is as simple as calling the evaluate method on the trainer object. This will run the model on the test set and compute the metrics we specified in the compute_metrics function.


```python
# Show the performance of the model on the test set
# What do you think the evaluation accuracy will be?
trainer.evaluate()
```








    {'eval_loss': 0.47647300362586975,
     'eval_accuracy': 0.792,
     'eval_runtime': 8.1044,
     'eval_samples_per_second': 61.695,
     'eval_steps_per_second': 15.424,
     'epoch': 1.0}



### View the results

Let's look at two examples with labels and predicted values.


```python
import pandas as pd

df = pd.DataFrame(tokenized_ds["test"])
df = df[["text", "label"]]

# Replace <br /> tags in the text with spaces
df["text"] = df["text"].str.replace("<br />", " ")

# Add the model predictions to the dataframe
predictions = trainer.predict(tokenized_ds["test"])
df["predicted_label"] = np.argmax(predictions[0], axis=1)

df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
      <th>predicted_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>When I unsuspectedly rented A Thousand Acres...</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is the latest entry in the long series of...</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Look at some of the incorrect predictions

Let's take a look at some of the incorrectly-predcted examples


```python
# Show full cell output
pd.set_option("display.max_colwidth", None)

df[df["label"] != df["predicted_label"]].head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
      <th>predicted_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>Don't pay any attention to the rave reviews of this film here. It is the worst Van Damme film and one of the worst of any sort I have ever seen. It would appeal to somebody with no depth whatever who requires nothing more than gunfire and explosions to be entertained.  Seeing that this is directed by Peter Hyams it has made me realise that Peter has no talent as a director, but is very good at filming explosions and the like. However, movies need other elements as well; for example, a story. This one didn't have one. This might explain the awfulness of some of Mr. Hyams' more recent films, hardly any better than this one, really.  One can't help wondering how some people ever were put behind a camera.</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Coming from Kiarostami, this art-house visual and sound exposition is a surprise. For a director known for his narratives and keen observation of humans, especially children, this excursion into minimalist cinematography begs for questions: Why did he do it? Was it to keep him busy during a vacation at the shore?   "Five, 5 Long Takes" consists of, you guessed it, five long takes. They are (the title names are my own and the times approximate):   "Driftwood and waves". The camera stands nearly still looking at a small piece of driftwood as it gets moved around by small waves splashing on a beach. Ten minutes.  "Watching people on the boardwalk". The camera stands still looking at the ocean horizon and a boardwalk. People walk across the camera frame, their faces too far and blurry to make them interesting. Eleven minutes.  "Six dogs at the water's edge". The camera stands still looking at the ocean horizon with a sandy stretch of beach nearby. Far away at the water's edge, six dogs not doing much, just relaxing. Sixteen minutes.  "Ducks in line, gaggle of ducks". The camera stands still looking at the ocean horizon near the water's edge. Dozen and dozen of ducks stream in single file from left to right. I assume that Kiarostami released them gradually. The last two ducks stop dead on their track and suddenly a gaggle of ducks rolls quietly from right to left. I assume Kiarostami collected the ducks and re-released all at the same time. It is not the first time that he deals with the contrast between organized and disorganized behavior. Eight minutes.  "Frog symphony, oops, I mean cacophony, for a stormy night". The camera stands over a pond at night. It's pitch black except for what appears to be the reflection of the moon on the undulating water. It is a stormy night and clouds race to cover the moon. The screen goes dark. What remains for us is the cacophony of frogs, howling dogs and, eventually, morning roosters. Hit me on the head if this was done in a single take. I saw this segment as a sound composition put together in the editing room and accompanied by a simple visualization. Twenty seven minutes!   Except for the mildly amusing ducks, this exercise in minimalism left me cold. A nonessential film for Kiarostami admirers.  I thought I would rate "Five" a five, but four is what it deserves.  The film is dedicated to Yasujiru Ozu.</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## End of exercise

Great work! 🤗

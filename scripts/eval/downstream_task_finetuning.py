import os
import sys
sys.path.append(os.path.abspath("."))
os.environ['WANDB_DISABLED'] = 'true'

from modules.custom_modeling_gpt2 import GPT2ForSentipolcClassification, GPT2ForSentipolcClassificationWithDropout, GPT2ForSequenceClassificationWithDropout
from modules.custom_modeling_bert import BertForSentipolcClassification
from transformers import (
    DataCollatorForTokenClassification,
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification,
    DataCollatorWithPadding,
    TrainingArguments, 
    AutoTokenizer, 
    Trainer,    
)

from datasets import load_dataset
from functools import partial
import numpy as np
import argparse
import evaluate
import ast


mae = evaluate.load('mae')
spearmanr = evaluate.load("spearmanr")
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
seqeval = evaluate.load("seqeval")

def compute_metrics_for_token_classification(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value
    final_results["precision"] = results["overall_precision"]
    final_results["recall"] = results["overall_recall"]
    final_results["f1"] = results["overall_f1"]
    final_results["accuracy"] = results["overall_accuracy"]
    return final_results
   

def compute_metrics_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    res = {
            'mae': mae.compute(predictions=logits, references=labels)['mae'],
            'spearmanr': spearmanr.compute(predictions=logits, references=labels)['spearmanr']
        }            
    return res


def compute_metrics_sentipolc_classification(eval_pred):
    logits, labels = eval_pred
    pos_labels = labels[0]
    pos_predictions = np.argmax(logits['pos'], axis=1)
    neg_labels = labels[1]
    neg_predictions = np.argmax(logits['neg'], axis=1)

    res = {
        'pos_accuracy': accuracy.compute(predictions=pos_predictions, references=pos_labels)['accuracy'],
        'neg_accuracy': accuracy.compute(predictions=neg_predictions, references=neg_labels)['accuracy'],
        'pos_f1': f1.compute(predictions=pos_predictions, references=pos_labels, average="weighted")['f1'],
        'neg_f1': f1.compute(predictions=neg_predictions, references=neg_labels, average="weighted")['f1']
    }


    return res


def compute_metrics_classification(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    res = {
        'accuracy': accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        'f1': f1.compute(predictions=predictions, references=labels, average="weighted")['f1'],
    }
    return res


TASKS_MAP = {
    'complexity': {'compute_metrics': compute_metrics_regression, 'num_labels': 1},
    'sentiment': {'compute_metrics': compute_metrics_sentipolc_classification, 'num_labels': 2},
    'coherence': {'compute_metrics': compute_metrics_classification, 'num_labels': 2},
    'pos_tagging': {'compute_metrics': compute_metrics_for_token_classification, 'num_labels': 235}
}

def get_model(model_path, task, num_labels):
    if task == 'sentiment':
        if 'bert' in model_path:
            model = BertForSentipolcClassification.from_pretrained(model_path, num_labels=num_labels)
        else: 
            model = GPT2ForSentipolcClassificationWithDropout.from_pretrained(model_path, num_labels=num_labels)
    elif task == 'pos_tagging':
        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels)
    else:
        if 'bert' in model_path:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        else:
            model = GPT2ForSequenceClassificationWithDropout.from_pretrained(model_path, num_labels=num_labels)

    return model


def get_gata_collator(tokenizer, task):
    if task == 'pos_tagging':
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator



def model_finetuning(model_path, downstream_task, num_labels, compute_metrics, output_path, epochs):
    if 'bert' in model_path:
        tokenizer_path = 'models/bert_tokenizer'
    else:
        tokenizer_path = 'models/gpt_tokenizer'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)

    model = get_model(model_path, downstream_task, num_labels)
    data_collator = get_gata_collator(tokenizer, downstream_task)
    tokenized_dataset = tokenize_dataset(downstream_task, tokenizer, model)

    # Forgot to set pad_token_id in the first pre-trained GPT model
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=output_path, 
        eval_strategy='epoch',
        logging_strategy='epoch',
        logging_dir=output_path,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        learning_rate=5e-05,
        weight_decay=0.01,
        warmup_ratio=0.01,
        save_strategy='no' 
        )

    if downstream_task == 'pos_tagging':
        compute_metrics = partial(compute_metrics, label_list=model.config.id2label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_state()

def tokenize_dataset(task, tokenizer, model):
    dataset = load_dataset_from_csv(task)
    if task == 'pos_tagging':
        tokenized_dataset = tokenize_dataset_dataset_for_token_classification(dataset, tokenizer, model)
    else:
        tokenized_dataset = tokenize_dataset_for_sentence_classification(dataset, tokenizer)
    return tokenized_dataset


def load_dataset_from_csv(task):
    data_files = {'train': f'data/evaluation/{task}/train.csv', 
                  'test': f'data/evaluation/{task}/test.csv'}
    
    dataset = load_dataset('csv', data_files=data_files)
    return dataset


def tokenize_dataset_for_sentence_classification(dataset, tokenizer):
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, desc="Running tokenizer on dataset")
    return tokenized_dataset

def convert_elements_to_list(example):
    example['text'] = ast.literal_eval(example['text'])
    example['labels'] = ast.literal_eval(example['label'])
    return example

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    label_list.append('UNK')
    return label_list

def tokenize_dataset_dataset_for_token_classification(dataset, tokenizer, model):
    dataset = dataset.map(convert_elements_to_list, remove_columns=['label'])
    
    label_list = get_label_list(dataset['train']['labels'])
    label_to_id = {l: i for i, l in enumerate(label_list)}

    model.config.label2id = label_to_id
    model.config.id2label = dict(enumerate(label_list))


    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['text'],
            padding=True,
            truncation=True,
            max_length=128,
            is_split_into_words=True,
            )
        labels = []
        for i, label in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_id = label_to_id[label[word_idx]] if label[word_idx] in label_to_id else label_to_id['UNK']
                    label_ids.append(label_id)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=['text','id'], desc="Running tokenizer on train dataset")
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str)
    parser.add_argument('-t', '--downstream_task', type=str, choices=['complexity', 'sentiment', 'coherence', 'pos_tagging'])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    args = parser.parse_args()

    if 'bert' in args.model_dir:
        model_name = args.model_dir.split('/')[-1] if 'bert' in args.model_dir.split('/')[-1] else args.model_dir.split('/')[-2]
    else:
        model_name = args.model_dir.split('/')[-1] if 'gpt' in args.model_dir.split('/')[-1] else args.model_dir.split('/')[-2]
    output_dir = os.path.join('models/downstream_tasks', args.downstream_task, model_name)
    task_map = TASKS_MAP[args.downstream_task]

    for checkpoint_name in os.listdir(args.model_dir):
        checkpoint_path = os.path.join(args.model_dir, checkpoint_name)
        if os.path.isdir(checkpoint_path):
            output_path = os.path.join(output_dir, checkpoint_name)
            model_finetuning(checkpoint_path, args.downstream_task, task_map['num_labels'], task_map['compute_metrics'], output_path, args.epochs)
    
    model_finetuning(args.model_dir, args.downstream_task, task_map['num_labels'], task_map['compute_metrics'], os.path.join(output_dir, 'checkpoint-117189'), args.epochs)


def main1():
    model_names = ['models/pretrained/bert_medium_42_train_1_gulpease', 'models/pretrained/bert_medium_42_train_1_gulpease_inverted', 'models/pretrained/bert_medium_42_train_1_orig_inverted']
    downstream_tasks = ['sentiment', 'complexity', 'pos_tagging']

    for downstream_task in downstream_tasks:
        for model_dir in model_names:
            model_name = model_dir.split('/')[-1] if 'bert' in model_dir.split('/')[-1] else model_dir.split('/')[-2]
            output_dir = os.path.join('models/downstream_tasks', downstream_task, model_name)
            task_map = TASKS_MAP[downstream_task]

            for checkpoint_name in os.listdir(model_dir):
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                if os.path.isdir(checkpoint_path):
                    output_path = os.path.join(output_dir, checkpoint_name)
                    if not os.path.exists(output_path):
                        model_finetuning(checkpoint_path, downstream_task, task_map['num_labels'], task_map['compute_metrics'], output_path, 10)
                    else:
                        print('skipping', checkpoint_path, 'on', downstream_task)

        if not os.path.exists(os.path.join(output_dir, 'checkpoint-117189')):   
            model_finetuning(model_dir, downstream_task, task_map['num_labels'], task_map['compute_metrics'], os.path.join(output_dir, 'checkpoint-117189'), 10)
        else:
            print('skipping final checkpoint on', downstream_task)


if __name__ == '__main__':
    main()


import os
import sys
sys.path.append(os.path.abspath("."))

from modules.custom_modeling_bert import BertForSentipolcClassification
from transformers import (
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    TrainingArguments, 
    AutoTokenizer, 
    Trainer,    
)

from datasets import load_dataset
import numpy as np
import argparse
import evaluate


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mae = evaluate.load('mae')
spearmanr = evaluate.load("spearmanr")
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


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
}


def model_finetuning(model_path, downstream_task, num_labels, compute_metrics, output_path, epochs):
    tokenizer_path = 'models/bert_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)

    tokenized_dataset = prepare_dataset(tokenizer, downstream_task)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if downstream_task == 'sentiment':
        model = BertForSentipolcClassification.from_pretrained(model_path, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

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

def prepare_dataset(tokenizer, task):
    data_files = {'train': f'data/evaluation/{task}/train.csv', 
                  'test': f'data/evaluation/{task}/test.csv'}
    
    dataset = load_dataset('csv', data_files=data_files)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, desc="Running tokenizer on dataset")
    return tokenized_dataset



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str)
    parser.add_argument('-t', '--downstream_task', type=str, choices=['complexity', 'sentiment', 'coherence'])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    args = parser.parse_args()

    model_name = args.model_dir.split('/')[-1]
    output_dir = os.path.join('models/downstream_tasks', args.downstream_task, model_name)
    task_map = TASKS_MAP[args.downstream_task]

    for checkpoint_name in os.listdir(args.model_dir):
        checkpoint_path = os.path.join(args.model_dir, checkpoint_name)
        if os.path.isdir(checkpoint_path):
            output_path = os.path.join(output_dir, checkpoint_name)
            model_finetuning(checkpoint_path, args.downstream_task, task_map['num_labels'], task_map['compute_metrics'], output_path, args.epochs)
    
    model_finetuning(args.model_dir, args.downstream_task, task_map['num_labels'], task_map['compute_metrics'], os.path.join(output_dir, 'checkpoint-117189'), args.epochs)

if __name__ == '__main__':
    main()
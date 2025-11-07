from transformers import AutoModel, AutoTokenizer, default_data_collator
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from utils import get_last_checkpoint
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import argparse
import shutil
import torch
import time
import json
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_tensor(tensor, out_dir):
    for layer_idx in range(tensor.shape[0]):
        out_path = os.path.join(out_dir, f'{layer_idx}.pt')
        torch.save(tensor[layer_idx], out_path)


def preprocess_dataset(dataset, tokenizer):
    def preprocessing_function(examples):
        result = tokenizer(examples['text'], padding='max_length', max_length=512, truncation=True)
        return result

    cols_to_remove = dataset.column_names
    cols_to_remove.remove('text')
    tokenized_dataset = dataset.map(
        preprocessing_function,
        batched=True,
        remove_columns=cols_to_remove,
        desc="Running tokenizer on dataset",
    )
    return tokenized_dataset


def extract_representations(model, dataloader, output_dir):
    all_hidden_states = None
    model.eval()
    with torch.no_grad():
        for batch_cpu in tqdm(dataloader):
            batch = {key: value.to(device) for key, value in batch_cpu.items()}
            hidden_states = model(**batch)['hidden_states']        
            non_pad_tokens = batch['attention_mask'].sum(axis=1)
            hidden_states = torch.stack(hidden_states, dim=1)

            mask = batch['attention_mask']
            mask = mask.view(mask.shape[0], 1, mask.shape[1], 1).expand(hidden_states.shape)
            masked_hidden_states = hidden_states * mask
            masked_hidden_states = masked_hidden_states[:, 1:, :, :]
            sum_hidden_states = torch.sum(masked_hidden_states, dim=2)
            average_hidden_states = torch.div(sum_hidden_states,
                                                non_pad_tokens.view(-1, 1, 1))  # batch_size, num_layers + 1, hidden_size
            average_hidden_states = average_hidden_states.transpose(0, 1)
            all_hidden_states = average_hidden_states if all_hidden_states is None else torch.cat(
                (all_hidden_states, average_hidden_states), dim=1)
            
        
        save_tensor(all_hidden_states, output_dir)


def save_dataset_representations(model_path, tokenizer_path, train_path, test_path, output_dir, batch_size):
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, return_tensors='pt')
    model.to(device)

    data_files = {'train': train_path, 'test':test_path}
    dataset = load_dataset('csv', data_files=data_files, sep='\t')
    train_dataset = preprocess_dataset(dataset['train'], tokenizer)
    test_dataset = preprocess_dataset(dataset['test'], tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator,
                                    batch_size=batch_size)

    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=default_data_collator,
                                    batch_size=batch_size)
    
    extract_representations(model, train_dataloader, os.path.join(output_dir, 'train'))
    extract_representations(model, test_dataloader, os.path.join(output_dir, 'test'))


def load_representations(src_path):
    return torch.load(src_path, map_location=torch.device('cpu'))

def load_dataframe(src_path):
    return pd.read_csv(src_path, sep='\t')

def load_labels(df, feature):
    return df[feature].tolist()

def save_predictions(out_path, predictions, labels):
     with open(out_path, 'w+') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        for y_pred, y_true in zip(predictions, labels):
            csv_writer.writerow([y_pred, y_true])


def already_coputed(src_dir):
    layer_0_dir = os.path.join(src_dir, '0')
    if os.path.exists(layer_0_dir):
        if len(os.listdir(layer_0_dir)) == 64:
            return True
    return False

def do_probing_tasks(model_path, output_dir, tokenizer_path, train_path='data/probing_data/train.tsv', test_path= 'data/probing_data/test.tsv', batch_size=16):
    if already_coputed(output_dir):
        print(f'Skipping {output_dir}')
        return
    
    os.makedirs(os.path.join(output_dir, 'train'))
    os.makedirs(os.path.join(output_dir, 'test'))

    save_dataset_representations(model_path, tokenizer_path, train_path, test_path, output_dir, batch_size)

    train_df = load_dataframe(train_path)
    test_df = load_dataframe(test_path)

    num_layers = len(os.listdir(os.path.join(output_dir, 'train')))

    features = [col_name for col_name in train_df.columns.tolist() if col_name not in ['identifier', 'text']]
    

    for layer in range(num_layers):
        os.mkdir(os.path.join(output_dir, str(layer)))
        train_path = os.path.join(output_dir, 'train', f'{layer}.pt')
        test_path = os.path.join(output_dir, 'test', f'{layer}.pt')
        train_sentences = load_representations(train_path)
        test_sentences = load_representations(test_path)    


        scaler = MinMaxScaler() 
        X_train = scaler.fit_transform(train_sentences)
        X_test = scaler.transform(test_sentences)

        for feat in features:
            train_labels = load_labels(train_df, feat)
            test_labels = load_labels(test_df, feat)
            regressor = Ridge()
            regressor.fit(X_train, train_labels)
            predictions = regressor.predict(X_test)

            save_predictions(os.path.join(output_dir, str(layer), f'{feat}.tsv'), predictions, test_labels)
    
    
    shutil.rmtree(os.path.join(output_dir, 'train'))
    shutil.rmtree(os.path.join(output_dir, 'test'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-t', '--train_path', type=str, default='data/probing_data/train.tsv')
    parser.add_argument('-e', '--test_path', type=str, default='data/probing_data/test.tsv')
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-k', '--tokenizer_path', type=str, default='models/bert_tokenizer')
    args = parser.parse_args()

    last_training_step = get_last_checkpoint(args.model_path)

    start_time = time.time()
    for checkpoint_dir in os.listdir(args.model_path):
        checkpoint_path = os.path.join(args.model_path, checkpoint_dir)
        model_str = '/'.join(args.model_path.split('/')[-2:])
        output_dir = os.path.join('data/probing_results', model_str, checkpoint_dir)
        if os.path.isdir(checkpoint_path):
            do_probing_tasks(checkpoint_path, output_dir, args.tokenizer_path, args.train_path, args.test_path, args.batch_size)

    do_probing_tasks(args.model_path, os.path.join('data/probing_results', model_str, f'checkpoint-{last_training_step}'), args.tokenizer_path,  args.train_path, args.test_path, args.batch_size)
    print("--- %s seconds ---" % (time.time() - start_time))
    

if __name__ == '__main__':
    main()

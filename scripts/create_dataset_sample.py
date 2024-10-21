import _pickle as pickle
from tqdm import tqdm
import numpy as np
import argparse
import random
import os

NUM_LINES = 768563962

def filter_dataset_sentences_by_length(src_dir: str, min_sent_len: int, max_sent_len: int) -> dict:
    print(f'Filtering dataset by sentence lenght ({min_sent_len} - {max_sent_len})\n')
    with tqdm(total=NUM_LINES) as pbar:
        ids_dict = dict()
        for file_name in os.listdir(src_dir):
            ids_dict[file_name] = []
            file_path = os.path.join(src_dir, file_name)
            for line in open(file_path, 'r'):
                if line.startswith('# sent_id = '):
                    sent_idx = line[len('# sent_id = '):].strip()
                    num_words = 0
                if line[0].isdigit():
                    word_idx = line.split('\t')[0]
                    if '-' not in word_idx:
                        num_words = int(word_idx)
                if line.strip() == '' and num_words >= min_sent_len and num_words <= max_sent_len:
                    ids_dict[file_name].append(int(sent_idx))
                pbar.update(1)
    return ids_dict

def load_sample_ids(ids_path: str, src_dir: str, min_sent_len: int, max_sent_len: int) -> dict:
    if not os.path.exists(ids_path):
        ids_dict = filter_dataset_sentences_by_length(src_dir, min_sent_len, max_sent_len)
        pickle.dump(ids_dict, open(ids_path, 'wb'))
    else:
        ids_dict = pickle.load(open(ids_path, 'rb'))
    return ids_dict

def create_ids_mask(ids_dict: dict, num_sentences: int) -> list:
    mask_len = sum([len(ids_list) for ids_list in ids_dict.values()])
    mask = [True]*num_sentences + [False]*(mask_len-num_sentences)
    random.shuffle(mask)
    return mask

def get_sample_ids(ids_dict:dict,  mask: list) -> dict:
    sample_ids = dict()
    last_mask_id = 0
    for doc_id, ids_list in ids_dict.items():
        mask_slice = mask[last_mask_id: last_mask_id + len(ids_list)]
        last_mask_id = last_mask_id + len(ids_list)
        sample_ids_list = np.array(ids_list)[np.array(mask_slice)]
        sample_ids[doc_id] = sample_ids_list.tolist()
    return sample_ids

def extract_dataset_sample(src_dir: str, sample_ids: dict, out_path:str):
    with tqdm(total=NUM_LINES) as pbar:
        with open(out_path, 'w') as out_file:
            for file_name in os.listdir(src_dir):
                copy = False
                for line in open(os.path.join(src_dir, file_name), 'r'): 
                    if line.startswith('# sent_id = '):
                        if len(sample_ids[file_name]) == 0:
                            break
                        copy = False
                        sent_idx = int(line[len('# sent_id = '):].strip())
                        line = f'# sent_id = {file_name}_{sent_idx}\n'
                        if sent_idx == sample_ids[file_name][0]:
                            sample_ids[file_name].pop(0)
                            copy = True
                    if copy:
                        out_file.write(line)
                    pbar.update(1)

def load_ids_to_exclude(samples_dir: str) -> dict:
    ids_files_paths = [os.path.join(samples_dir, file_name) for file_name in os.listdir(samples_dir) if file_name.endswith('ids.pkl')]
    if len(ids_files_paths) == 0:
        return None
    print('Filtering ids of the following samples:', ','.join([file_path.split('_')[-2] for file_path in ids_files_paths]))
    ids_to_exclude = dict()
    for file_path in ids_files_paths:
        sample_ids = pickle.load(open(file_path, 'rb'))
        for file_name in sample_ids.keys():
            if file_name not in ids_to_exclude:
                ids_to_exclude[file_name] = []
            ids_to_exclude[file_name] += sample_ids[file_name]
    return ids_to_exclude
            

def filter_ids_dict(ids_dict: dict, ids_to_exclude: dict) -> dict:
    if ids_to_exclude is not None:
        for file_name in ids_dict.keys():
            ids_dict[file_name] = sorted(list(set(ids_dict[file_name]) - set(ids_to_exclude[file_name])))
    return ids_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--min_sentence_length', type=int, default=5)
    parser.add_argument('-u', '--max_sentence_length', type=int, default=60)
    parser.add_argument('-n', '--num_sentences', type=int, default=10000000)
    parser.add_argument('-i', '--sample_idx', required=True)
    args = parser.parse_args()
    
    random.seed(42)

    src_dir = '/home/luca/Workspace/wiki_ita/wiki_parsed'
    filtered_ids_path =  f'data/ids_min_len_{args.min_sentence_length}_max_len_{args.max_sentence_length}.pkl'

    samples_dir = 'data/dataset_samples/'
    out_path = os.path.join(samples_dir, f'sample_{args.sample_idx}.conllu')
    sample_ids_path = os.path.join(samples_dir, f'sample_{args.sample_idx}_ids.pkl')

    if os.path.exists(out_path):
        raise Exception(f'Sample {args.sample_idx} already exists.')

    ids_dict = load_sample_ids(filtered_ids_path, src_dir, args.min_sentence_length, args.max_sentence_length)
    other_samples_ids = load_ids_to_exclude(samples_dir)
    ids_dict = filter_ids_dict(ids_dict, other_samples_ids)
    mask = create_ids_mask(ids_dict, args.num_sentences)
    sample_ids = get_sample_ids(ids_dict, mask)
    pickle.dump(sample_ids, open(sample_ids_path, 'wb'))
    extract_dataset_sample(src_dir, sample_ids, out_path)
    

if __name__ == '__main__':
    main()

import _pickle as pickle
from tqdm import tqdm
import requests
import csv
import os

def load_all_ids(src_path):
    ids_dict = pickle.load(open(src_path, 'rb'))
    all_ids = []
    for doc_id in ids_dict.keys():
        for sent_idx in ids_dict[doc_id]:
            all_ids.append(f'{doc_id}_{sent_idx}')
    return all_ids

def load_annotated_and_error_ids(src_path):
    error_ids = []
    annotated_ids = []
    with open(src_path, 'r') as src_file:
        csv_reader = csv.reader(src_file)
        for row in csv_reader:
            sent_id = row[0]
            annotated_ids.append(sent_id)
            scores = [float(score) for score in row[2:]]
            if any(el > 100 for el in scores):
                error_ids.append(sent_id)
    return annotated_ids, error_ids

def load_second_pass_processed_ids(src_path):
    annotated_ids = []
    with open(src_path, 'r') as src_file:
        csv_reader = csv.reader(src_file)
        for row in csv_reader:
            sent_id = row[0]
            annotated_ids.append(sent_id)
    return annotated_ids

def load_ids_to_annotate(all_ids_path, annotated_path, out_path, ids_to_skip=None):
    all_ids = load_all_ids(all_ids_path)
    annotated_ids, error_ids = load_annotated_and_error_ids(annotated_path)

    ids_to_annotate = list(set(all_ids) - set(annotated_ids))

    if os.path.exists(out_path):
        second_pass_ids = load_second_pass_processed_ids(out_path)
        ids_to_annotate = list(set(ids_to_annotate) - set(second_pass_ids))

    if ids_to_skip is not None:
        ids_to_annotate = list(set(ids_to_annotate) - set(ids_to_skip))
    return ids_to_annotate + error_ids

def load_sentences(src_path,):
    sentences_dict = dict()
    with open(src_path, 'r') as src_file:
        csv_reader = csv.reader(src_file)
        for row in csv_reader:
            sent_id = row[0]
            sentences_dict[sent_id] = row[1]
    return sentences_dict

def compute_readit_score(sentence):
    SERVER_PATH =  "http://localhost:1200"

    def load_document(text):
        r = requests.post(SERVER_PATH + '/documents/',           # carica il documento nel database del server
                        data={'text': text,                    # durante il caricamento viene eseguita un'analisi linguistica necessaria per calcolare la leggibilita'
                            'lang': 'IT',
                            'extra_tasks': ["readability"]     # chiede al server di calcolare anche la leggibilità del docuemnto
                    })
        doc_id = r.json()['id']                                  # id del documento nel database del server, che serve per richiedere i risultati delle analisi
        return doc_id
    
    def get_readability_scores(doc_id):
        r = requests.get(SERVER_PATH + '/documents/details/%s' % doc_id)
        sent_readability = {'base': [], 'lexical': [], 'syntax': [], 'all': []}
        for sent_results in r.json()['sentences']['data']:
            if sent_readability['base'] is not None:
                sent_readability['base'].append(sent_results['readability_score_base'])
                sent_readability['lexical'].append(sent_results['readability_score_lexical'])
                sent_readability['syntax'].append(sent_results['readability_score_syntax'])
                sent_readability['all'].append(sent_results['readability_score_all'])
        return sent_readability
    
    def compute_average_scores(sentence_scores):
        average_scores = dict()
        for r_type in sentence_scores.keys():
            not_none_scores = [score for score in sentence_scores[r_type] if score is not None]
            if len(not_none_scores) > 0:
                average_scores[r_type] = sum(not_none_scores)/len(not_none_scores)
            else:
                average_scores[r_type] = 999
        return average_scores
    
    try:
        doc_id = load_document(sentence)
    except:
        print(f'Post Error on Sentence: {sentence}')
        return None
    sent_readability = get_readability_scores(doc_id)
    result = compute_average_scores = compute_average_scores(sent_readability)

    return result    

def main():
    annotated_path = 'data/dataset_samples/sample_1_readit.csv'
    all_ids_path = 'data/dataset_samples/sample_1_ids.pkl'
    sentences_path = 'data/dataset_samples/sample_1_sentences.csv'
    out_path = 'data/dataset_samples/sample_1_readit_second_pass.csv'

    ids_to_annotate = load_ids_to_annotate(all_ids_path, annotated_path, out_path)
    sentences_dict = load_sentences(sentences_path)

    with open(out_path, 'a') as out_file:
        csv_writer = csv.writer(out_file)
        for sent_idx, sent_id in tqdm(enumerate(ids_to_annotate), total=len(ids_to_annotate)):
            sentence = sentences_dict[sent_id]
            result = compute_readit_score(sentence)
            if result is not None:
                complexities = [result[comp_type] for comp_type in ['base', 'lexical', 'syntax', 'all']]
                csv_writer.writerow([sent_id, sentence]+complexities)

if __name__ == '__main__':
    main()
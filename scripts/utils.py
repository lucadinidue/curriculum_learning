import pandas as pd
import csv

def write_sentences_to_file(sentences:list, out_path:str, ordered_keys:list|None=None):
    with open(out_path, 'a') as out_file:
        csv_writer = csv.writer(out_file, delimiter=',')
        for sentence in sentences:
            if type(sentence.complexity) == dict:
                complexities = [sentence.complexity[key] for key in ordered_keys]
                csv_writer.writerow([sentence.sentence_id, sentence.text] + complexities)
            else:    
                csv_writer.writerow([sentence.sentence_id, sentence.text, sentence.complexity])

def load_dataset_from_csv(src_path:str) -> pd.DataFrame:
    df = pd.read_csv(src_path, names=['sent_id', 'text', 'complexity'])
    return df
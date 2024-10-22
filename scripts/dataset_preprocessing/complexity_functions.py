from torch.nn import CrossEntropyLoss
from classes import Sentence
from evaluate import load
from tqdm import tqdm
import requests
import torch
import time


def compute_sentence_length(sentences:list[Sentence]):
    for sentence in sentences:
        token_ids = [token.token_id for token in sentence.tokens]
        last_token_id = token_ids[-1]#.split('\t)[0]
        sentence.set_complexity(int(last_token_id))
        sentence.delete_tokens()


# adapted from Huggingface: https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py
def compute_model_perplexity(sentences:list[Sentence], **kwargs): #model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_length:None|int=None):
    
    tokenizer = kwargs['tokenizer']
    model = kwargs['model']
    max_length = None

    texts = [sentence.text for sentence in sentences]
    batch_size = len(sentences)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # add padding token to tokenizer
    existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
    tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})


    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_length else False,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    try:
        assert torch.all( torch.ge(attn_masks.sum(1), 2)), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."
    except AssertionError:
        for sentence in sentences:
            sentence.set_complexity(None)
            sentence.delete_tokens()
        return

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    computed = False
    while not computed:
        computed = True
        try:
            for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
                end_index = min(start_index + batch_size, len(encoded_texts))
                encoded_batch = encoded_texts[start_index:end_index]
                attn_mask = attn_masks[start_index:end_index]

                labels = encoded_batch

                with torch.no_grad():
                    out_logits = model(encoded_batch, attention_mask=attn_mask).logits

                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

                perplexity_batch = torch.exp(
                    (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                    / shift_attention_mask_batch.sum(1)
                )

                ppls += perplexity_batch.tolist()

            for sentence, sentence_perplexity in zip(sentences, ppls):
                sentence.set_complexity(sentence_perplexity)
                sentence.delete_tokens()

        except torch.cuda.OutOfMemoryError:
            batch_size = int(batch_size / 2)
            computed = False
            if batch_size == 0:
                for sentence in sentences:
                    sentence.set_complexity(None)
                    sentence.delete_tokens()
                return
            


def compute_model_perplexity_old(sentences:list[Sentence], model_name:str='local_models/Minerva-350M-base-v1.0'):
    texts = [sentence.text for sentence in sentences]
    batch_size = len(texts)
    perplexity = load('perplexity', module_type='metric')
    computed = False
    while not computed:
        computed = True
        try:
            model_perplexities = perplexity.compute(model_id=model_name, add_start_token=False, predictions=texts, batch_size=batch_size, max_length=16384)
            for sentence, sentence_perplexity in zip(sentences, model_perplexities['perplexities']):
                sentence.complexity = sentence_perplexity
                sentence.delete_tokens()
        except torch.cuda.OutOfMemoryError:
            batch_size = int(batch_size / 2)
            computed = False


def compute_gulpease_index(sentences:list[Sentence]):
    for sentence in sentences:
        try:
            sentence.complexity = 89 + (300 - 10*sentence.get_num_chars()) / sentence.get_num_words()
        except:
            sentence.complexity = 99999999999
        sentence.delete_tokens()

def compute_readit_score(sentences:list[Sentence]):
    SERVER_PATH =  "http://localhost:1200" #"http://api.italianlp.it"
    readability_types = ['base', 'lexical', 'syntax', 'all']


    def load_document(text):
        r = requests.post(SERVER_PATH + '/documents/',           # carica il documento nel database del server
                        data={'text': text,                    # durante il caricamento viene eseguita un'analisi linguistica necessaria per calcolare la leggibilita'
                            'lang': 'IT',
                            'async': True,                      # chiamata asincrona per gestire testi più grandi
                            'extra_tasks': ["readability"]     # chiede al server di calcolare anche la leggibilità del docuemnto
                    })
        doc_id = r.json()['id']                                  # id del documento nel database del server, che serve per richiedere i risultati delle analisi
        return doc_id
    
    def wait_for_readability_completed(doc_id):
        completed = False
        pages_results = []
        page = 1
        while not completed:
            result = requests.get(SERVER_PATH + '/documents/details/%s?page=%s' % (doc_id, page))
            json_value = result.json()
            if json_value['readability_executed']:
                pages_results += json_value['sentences']['data']
                if json_value['sentences']['next']:
                    page += 1
                else:
                    completed = True
            else:
                time.sleep(0.01)
        return pages_results

    def get_readability_scores(result):
        all_sent_readability = list()
        for sent_results in result: #['sentences']['data']:
            sent_readability = dict()
            text = sent_results['raw_text']
            sent_readability['text'] = text
            sent_readability['base'] = sent_results['readability_score_base']
            sent_readability['lexical'] = sent_results['readability_score_lexical']
            sent_readability['syntax'] = sent_results['readability_score_syntax']
            sent_readability['all'] = sent_results['readability_score_all']
            all_sent_readability.append(sent_readability)
        return all_sent_readability

    def parse_results(readability_scores):
        parsed_result = dict()
        sent_id = None
        for sent_dict in readability_scores:
            if sent_dict['text'] is None:
                print('Frase vuota')
                print(sent_dict)
            elif sent_dict['text'].startswith('SENTENCE_wiki_'):
                sent_id = sent_dict['text'][len('SENTENCE_'):]
                parsed_result[sent_id] = {r_type: [] for r_type in readability_types}
            else:
                if sent_id is not None:
                    for r_type in readability_types:
                        parsed_result[sent_id][r_type].append(sent_dict[r_type])
                else:
                    print('Errore')
                    print(sent_dict)
        return parsed_result

    def compute_average_scores(sentence_scores):
        average_scores = dict()
        for r_type in sentence_scores.keys():
            if len(sentence_scores[r_type]) > 0:
                average_scores[r_type] = sum(sentence_scores[r_type])/len(sentence_scores[r_type])
            else:
                average_scores[r_type] = 999
        return average_scores
    
    def remove_none_scores(sentence_scores):
        cleaned_scores = dict()
        for r_type in sentence_scores.keys():
            cleaned_scores[r_type] = [score for score in sentence_scores[r_type] if score is not None]
        return cleaned_scores

    text = ''      
    for sentence in sentences:
        text += f'SENTENCE_{sentence.sentence_id}\n\n'
        text += sentence.text+'\n\n'
        sentence.delete_tokens()
    
    doc_id = load_document(text)
    result = wait_for_readability_completed(doc_id)
    readability_scores = get_readability_scores(result)
    parsed_result = parse_results(readability_scores)

    for sentence in sentences:
        if sentence.sentence_id in parsed_result:
            sentence_scores = parsed_result[sentence.sentence_id]
            try:
                sentence.complexity = compute_average_scores(sentence_scores)
            except:
                sentence_scores = remove_none_scores(sentence_scores)
                sentence.complexity = compute_average_scores(sentence_scores)
        else:
            sentence.complexity = {r_type: 888 for r_type in readability_types}


def compute_readit_score_old(sentences:list[Sentence]):
    SERVER_PATH =  "http://localhost:1200" #"http://api.italianlp.it"

    def load_document(text):
        r = requests.post(SERVER_PATH + '/documents/',           # carica il documento nel database del server
                        data={'text': text,                    # durante il caricamento viene eseguita un'analisi linguistica necessaria per calcolare la leggibilita'
                            'lang': 'IT',
                            'extra_tasks': ["readability"]     # chiede al server di calcolare anche la leggibilità del docuemnto
                    })
        doc_id = r.json()['id']                                  # id del documento nel database del server, che serve per richiedere i risultati delle analisi
        return doc_id
    

    def compute_average_scores(sentence_scores):
        average_scores = dict()
        for r_type in sentence_scores.keys():
            if len(sentence_scores[r_type]) > 0:
                average_scores[r_type] = sum(sentence_scores[r_type])/len(sentence_scores[r_type])
            else:
                average_scores[r_type] = 999
        return average_scores


    def get_readability_scores(doc_id):
        r = requests.get(SERVER_PATH + '/documents/details/%s' % doc_id)
        sent_readability = {'base': [], 'lexical': [], 'syntax': [], 'all': []}
        for sent_results in r.json()['sentences']['data']:
            if sent_readability['base'] is not None:
                sent_readability['base'].append(sent_results['readability_score_base'])
                sent_readability['lexical'].append(sent_results['readability_score_lexical'])
                sent_readability['syntax'].append(sent_results['readability_score_syntax'])
                sent_readability['all'].append(sent_results['readability_score_all'])
        return compute_average_scores(sent_readability)        
            
            
    for sentence in sentences:
        try:
            api_id = load_document(sentence.text)
            sentence.complexity = get_readability_scores(api_id)
        except:
            sentence.complexity = {'base': 888, 'lexical': 888, 'syntax': 888, 'all': 888}
        sentence.delete_tokens()

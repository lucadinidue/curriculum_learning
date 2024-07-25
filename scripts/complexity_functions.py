from classes import Sentence
from evaluate import load
import torch

def compute_sentence_length(sentences:list[Sentence]):
    for sentence in sentences:
        token_ids = [token.token_id for token in sentence.tokens]
        last_token_id = token_ids[-1].split('\t')[0]
        sentence.comlpexity = int(last_token_id)
        sentence.delete_tokens()


def compute_model_perplexity(sentences:list[Sentence], model_name:str='openai-community/gpt2'):
    texts = [sentence.text for sentence in sentences]
    batch_size = len(texts)
    perplexity = load('perplexity', module_type='metric')
    computed = False
    while not computed:
        computed = True
        try:
            model_perplexities = perplexity.compute(model_id=model_name, add_start_token=False, predictions=texts, batch_size=batch_size, max_length=1024)
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

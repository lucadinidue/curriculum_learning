from evaluate import load


def compute_sentence_length(sentences:list):
    for sentence in sentences:
        token_ids = [token.token_id for token in sentence.tokens]
        last_token_id = token_ids[-1].split('\t')[0]
        sentence.comlpexity = int(last_token_id)
        sentence.delete_tokens()


def compute_model_perplexity(sentences:list, model_name:str='openai-community/gpt2'):
    texts = [sentence.text for sentence in sentences]
    batch_size = len(texts)
    perplexity = load('perplexity', module_type='metric')
    model_perplexities = perplexity.compute(model_id=model_name, add_start_token=False, predictions=texts, batch_size=batch_size)
    for sentence, sentence_perplexity in zip(sentences, model_perplexities['perplexities']):
        sentence.complexity = sentence_perplexity
        sentence.delete_tokens()

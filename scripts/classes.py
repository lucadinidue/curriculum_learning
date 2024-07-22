class Token():

    def __init__(self, token_line:str):
        splitted_token = token_line.strip().split('\t')
        self.token_id = splitted_token[0]
        self.form = splitted_token[1]
        self.lemma = splitted_token[2]
        self.upos = splitted_token[3]
        self.xpos = splitted_token[4]
        self.feats = splitted_token[5]
        self.head = splitted_token[6]
        self.deprel = splitted_token[7]
        self.deps = splitted_token[8]
        self.misc = splitted_token[9]
        

class Sentence():

    def __init__(self, sentence_id:str):
        self.sentence_id = sentence_id
        self.text = None
        self.tokens = []
        self.complexity = None

    def add_token(self, token:Token):
        pass
        #self.tokens.append(token)

    def set_text(self, text:str):
        self.text = text

    def set_complexity(self, complexity_score:float):
        self.complexity = complexity_score

    def delete_tokens(self):
        self.tokens = None

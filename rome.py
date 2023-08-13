import torch
from args import get_args
from components.model import ScorerNN
from components.grammar import Grammar
from components.emd import EMD_mod
from components.ted_se import TEDse
from unidecode import unidecode
import re

class RoMe:

    def __init__(self):
        self.args = get_args()
        self.emd = EMD_mod(self.args)      # semantic similarity
        self.tedse = TEDse(self.args)      # semantic enhanced tree edit distance
        self.grammar = Grammar()  # grammatical acceptability
        self.network = ScorerNN()
        self.network.load_state_dict(torch.load("saved_model/checkpoint.pth"))
        self.network.eval()

    def clean_str(self, text):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        text = unidecode(text)
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\"", "", text)
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation marks
        text = re.sub("\s\s+", " ", text)  # remove multiple spaces within the text
        return text.strip().lower()

    def evaluate(self, hyp=None, ref=None):
        # cleaning the text and removing punctuation marks
        hyp = self.clean_str(hyp)
        ref = self.clean_str(ref)

        g_score = self.grammar.check_grammar(hyp)
        t_score = self.tedse.semantic_ted(ref, hyp)
        r_score = self.emd.eval(gold_sent=ref,hyp_sent=hyp)

        inp_x = [g_score, t_score, r_score]
        inp_x = torch.Tensor(inp_x).unsqueeze(0)
        pred = self.network(inp_x)
        score = torch.sigmoid(pred)
        return score.item()


if __name__=="__main__":

    hyp = "Here is a quote from Leonardo DiCaprio:"
    ref = "Here is a quote from Leonardo DiCaprio: \"To me, being an actor is not about being famous, it's about being human.\""

    rome = RoMe()
    score = rome.evaluate(ref,hyp)
    print(score)

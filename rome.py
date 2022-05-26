import torch
from args import get_args
from components.model import ScorerNN
from components.grammar import Grammar
from components.emd import EMD_mod
from components.ted_se import TEDse


class RoMe:

    def __init__(self):
        self.args = get_args()
        self.emd = EMD_mod(self.args)      # semantic similarity
        self.tedse = TEDse(self.args)      # semantic enhanced tree edit distance
        self.grammar = Grammar()  # grammatical acceptability
        self.network = ScorerNN()
        self.network.load_state_dict(torch.load("saved_model/checkpoint.pth"))
        self.network.eval()

    def evaluate(self, hyp=None, ref=None):
        g_score = self.grammar.check_grammar(hyp)
        t_score = self.tedse.semantic_ted(ref, hyp)
        r_score = self.emd.eval(gold_sent=ref,hyp_sent=hyp)

        inp_x = [g_score, t_score, r_score]
        inp_x = torch.Tensor(inp_x).unsqueeze(0)
        pred = self.network(inp_x)
        score = torch.sigmoid(pred)
        return score.item()

if __name__=="__main__":

    hyp = "103 hera discoverer james craig watson james craig watson deathcause peritonitis"
    ref = "james craig watson , who died of peritonitis , was the discoverer of 103 hera ."

    rome = RoMe()
    score = rome.evaluate(ref,hyp)
    print(score)

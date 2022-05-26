# part of the code is adopted from https://github.com/AIPHES/emnlp19-moverscore/blob/master/moverscore_v2.py

from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import os, torch
import pulp
import string
from transformers import AlbertModel, AlbertConfig, AlbertTokenizer

# setting environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(101)
np.random.seed(101)
np.seterr(over='raise')


class EMD_mod:
    def __init__(self, args=None, gamma= 0.07, delta=0.6):
        if args:
            self.args = args
            self.model_name = self.args.emdlm
            self.delta = self.args.deltaemd
            self.gamma = self.args.gammaval
        else:
            self.model_name = "albert-large-v2"
            self.gamma = gamma
            self.delta = delta

        self.config = AlbertConfig.from_pretrained(self.model_name, output_hidden_states=True, output_attentions=True)
        self.tokenizer = AlbertTokenizer.from_pretrained(self.model_name, do_lower_case=True)
        self.model = AlbertModel.from_pretrained(self.model_name, config=self.config)

    def get_weights_ref(self, references):
        """
        :param references: reference sentences
        :return: Update frequency and weights of words of reference sentences
        """
        word_freq_ref = self.get_word_frequencies(references)
        doc_freq_ref = self.get_doc_frequencies(references, word_freq_ref)
        weights_ref = [[self.get_weights(w, word_freq_ref, doc_freq_ref, len(references)) for w in sent.split()]
                       for sent in references]
        weights_ref = [self.calc_normalized_weight(weight) for weight in weights_ref]

        return weights_ref


    def truncate(self, tokens):
        """
        Truncate tokens more than the model length
        """
        if len(tokens) > self.tokenizer.model_max_length - 2:
            tokens = tokens[0:(self.tokenizer.model_max_length - 2)]
        return tokens


    def padding(self, arr, pad_token, dtype=torch.long):
        """
        Pad by maximum length
        """
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens, mask


    def collate_fn(self,arr, tokenize, numericalize, pad="[PAD]"):
        tokens = [["[CLS]"] + self.truncate(tokenize(a)) + ["[SEP]"] for a in arr]
        arr = [numericalize(a) for a in tokens]
        pad_token = numericalize([pad])[0]
        padded, lens, mask = self.padding(arr, pad_token, dtype=torch.long)
        return padded, lens, mask, tokens


    def get_embedding(self, all_sens, batch_size=-1):
        padded_sens, lens, mask, tokens = self.collate_fn(all_sens, self.tokenizer.tokenize, self.tokenizer.convert_tokens_to_ids)

        if batch_size == -1:
            batch_size = len(all_sens)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                self.model.eval()
                with torch.no_grad():
                    _,_,batch_embedding,_ = self.model(padded_sens[i:i + batch_size], attention_mask=mask[i:i + batch_size])
                    batch_embedding = torch.stack(batch_embedding)
                    embeddings.append(batch_embedding)
                    del batch_embedding

        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, lens, mask, tokens


    def pos_inf(self, pos_ti, pos_ri, m, n):
        """
        Computing word position information
        """
        pos_i = abs(pos_ti/(float(m) + 1e-30) - (pos_ri/(float(n) + 1e-30)))
        return pos_i

    def align_score(self, pos_pred, pos_gold, sent_pred, sent_gold, ref_emb=None, hyp_emb=None):
        n = len(sent_gold)
        m = len(sent_pred)
        return np.multiply(cosine_similarity(hyp_emb[pos_pred-1], ref_emb[pos_gold-1])[0][0],(1.0 - self.pos_inf(pos_pred, pos_gold, m, n)))

    @staticmethod
    def calc_normalized_weight(weight):
        total = np.sum(weight)
        wt = [np.divide(w, total) for w in weight]
        return wt

    def check_alignment(self, vec):
        for itm in vec:
            if itm < 1.0:
                return True
        return False

    def get_distance_matrix(self, inputs, references, ref_emb=None, hyp_emb=None):
        """
        Computing the distance matrix for the EMD
        """
        idx, sent_pred = inputs
        sent_gold = references[idx]
        sent_pred = sent_pred.split()
        sent_gold = sent_gold.split()
        m = len(sent_pred)
        n = len(sent_gold)

        d = np.ones((m, n))
        aligned_matrix = np.zeros((m, n))

        for i, word_pred in enumerate(sent_pred):
            for j, word_gold in enumerate(sent_gold):
                aligned_matrix[i][j] = self.align_score(i+1,j+1, sent_pred, sent_gold, ref_emb, hyp_emb)

        try:
            mx_idx = np.argmax(aligned_matrix, axis=1)
        except:    # For empty sequences
            mx_idx = np.zeros(len(aligned_matrix),dtype=int)

        for i, id in enumerate(mx_idx):
            simvalue = cosine_similarity(ref_emb[id], hyp_emb[i])[0][0]
            if simvalue > self.delta and not self.check_alignment(d[:,id]):
                adjust = abs(m - n + 1e-30) * 2 / (m + n + 1e-30)
                d[i][id] = 1.0 - (simvalue * np.exp(-adjust*self.gamma*self.pos_inf(i + 1, id + 1, m, n)))

        return d

    def get_weights(self, word, word_freq, doc_freq, N):
        """
        Computes weight of a word
        """
        if doc_freq[word]!=0:
            return (1+np.log10(word_freq[word])) * (np.log10(1 + (N/doc_freq[word])))
        else:
            return 0.0

    def get_word_frequencies(self, sents):
        """
        Computes word frequencies of words of a sentence
        """
        word_freq = defaultdict(float)
        for sent in sents:
            for w in sent.split():
                word_freq[w] += 1.0
        return word_freq

    def get_doc_frequencies(self, sents, word_freq):
        '''
        calculate doc frequencies
        :param sents:
        :param word_freq:
        :return:
        '''
        doc_freq = defaultdict(float)
        for w in word_freq:
            for sent in sents:
                if w in sent.split():
                    doc_freq[w] += 1.0
        return doc_freq

    def min_EMD_WORK_parallel(self, wN, wP, distance_matrix=None):
        """
        Calculates EMD using PulP
        :return:
        """
        #print (id)
        d = distance_matrix
        wp, wn = wP, wN
        m, n = len(wp), len(wn)

        # Initialize the model
        model = pulp.LpProblem("Earth_Mover_Distance", pulp.LpMinimize)

        # Define Variable
        F = pulp.LpVariable.dicts("F", ((i, j) for i in range(m) for j in range(n)), lowBound=0, cat='Continuous')

        # Define objective function
        model += sum([d[i][j] * F[i, j] for i in range(m) for j in range(n)])  # Minimizing the WORK

        wpsum = np.sum(wp)
        wnsum = np.sum(wn)

        # constraint #1
        for i, _ in enumerate(wp):
            for j, _ in enumerate(wn):
                model += F[i, j] >= 0

        # constraint #2
        for j, _ in enumerate(wn):
            for i, _ in enumerate(wp):
                model += F[i, j] <= wp[i]

        # constraint #3
        for i, _ in enumerate(wp):
            for j, _ in enumerate(wn):
                model += F[i, j] <= wn[j]

        # constraint #4
        sumFij = min(wpsum, wnsum)
        model += pulp.lpSum(F) == sumFij

        try:
            model.solve(pulp.PULP_CBC_CMD(msg=False))
            return pulp.value(model.objective), sumFij
        except Exception as e:
            return 1, sumFij


    def compute(self, references, sents_pred, ref_emb=None, hyp_emb=None, weights_ref=None):
        assert(len(references)==len(sents_pred))
        if sents_pred[0]=="" or references[0]=="":
            return 0.0

        word_freq_trans = self.get_word_frequencies(sents_pred)
        doc_freq_trans = self.get_doc_frequencies(sents_pred, word_freq_trans)

        weights_trans = [[self.get_weights(w, word_freq_trans, doc_freq_trans, len(sents_pred)) for w in sent.split()] for sent in sents_pred]
        weights_trans = [self.calc_normalized_weight(weight) for weight in weights_trans]
        dist_matrices = [self.get_distance_matrix(idxs, references, ref_emb, hyp_emb) for idxs in enumerate(sents_pred)]

        min_emd = [self.min_EMD_WORK_parallel(weights_ref[i], weights_trans[i], dist_matrices[i]) for i,_ in enumerate(weights_trans)]
        romescore = [(1.0 - np.divide(m_w, float(tot_fij))) if m_w is not None else 0 for m_w, tot_fij in min_emd]
        return np.average(romescore)


    def update_embeddings(self, sentences):
        """
        Extract embedding of each of the words in a sentence and update the embedding in the global variable
        :param sentences: A list of sentences
        :param doc_type: type of the sentence. choices: hyp, ref
        """
        temp_embedding, _, _, ref_tokens = self.get_embedding(sentences)
        temp_ids = [k for k, w in enumerate(ref_tokens[0]) if not w.startswith("▁")]
        temp_embedding = temp_embedding[-1]
        temp_embedding[0, temp_ids, :] = 0
        temp_embedding = temp_embedding[0][1:len(sentences[0].split()) + 1]

        return [emb.reshape(1, -1) for emb in temp_embedding]

    def clean_string(self, gold_sent, hyp_sent):
        gold_sent = [" ".join([w for w in sent.split() if w not in set(string.punctuation)]).strip() for sent in gold_sent]
        hyp_sent = [" ".join([w for w in sent.split() if w not in set(string.punctuation)]).strip() for sent in hyp_sent]

        for punct in set(string.punctuation):
            gold_sent = [gsent.replace(punct,"") for gsent in gold_sent]
            hyp_sent = [hsent.replace(punct,"") for hsent in hyp_sent]

        return gold_sent, hyp_sent


    def eval(self, gold_sent, hyp_sent):
        if isinstance(gold_sent,str):
            gold_sent = [gold_sent]
        if isinstance(hyp_sent,str):
            hyp_sent = [hyp_sent]

        gold_sent, hyp_sent = self.clean_string(gold_sent, hyp_sent)

        gold_sent = [gs.lower() for gs in gold_sent]
        hyp_sent = [hs.lower() for hs in hyp_sent]
        ref_emb = self.update_embeddings(gold_sent)
        hyp_emb = self.update_embeddings(hyp_sent)

        weights_ref = self.get_weights_ref(gold_sent)
        result = self.compute(gold_sent, hyp_sent, ref_emb=ref_emb, hyp_emb=hyp_emb, weights_ref=weights_ref)
        return round(result, 4)


if __name__ == "__main__":

    ref = ["Germany is situated in the central of europe"]
    hyp = ["Germany is positioned in the center of europe"]

    evaluator = EMD_mod()
    print(evaluator.eval(ref, hyp))
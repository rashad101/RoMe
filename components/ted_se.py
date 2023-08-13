import stanza
import edist.ted as ted
import string
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import fasttext
import re

class TEDse:
    def __init__(self, delta= 0.65, args=None):
        if args:
            self.args = args
            self.deltated = self.args.deltated
        else:
            self.deltated = delta

        # initialize stanza english pipeline
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', depparse_pretagged=True)

        # loading embedding
        print("Loading Fasttext embedding...")
        self.vec = fasttext.load_facebook_vectors("data/wiki.simple.bin")

    def stanza2ted(self, text, key=None):
        """
        Converts stanza dependency tree into incidence matrix to match with the edist format
        :param key -> token or lemma
        :param type -> sentence type hyp or ref
        """
        adjacency_list = []

        text = text.translate(str.maketrans('', '', string.punctuation))
        doc = self.nlp(text)

        # extract dependencies for each token from stanza document
        dependency_pairs = {x: [] for x in range(0, doc.num_tokens)}
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.head != 0:
                    dependency_pairs[word.head - 1].append(int(word.id) - 1)

        vertices = [word.text if key=="token" else word.lemma for word in sentence.words for sentence in doc.sentences]

        for k, v in dependency_pairs.items():
            adjacency_list.insert(k, v)

        return vertices, adjacency_list


    def diff(self, x,y, deltaed=0.65):
        """
        :param x: reference tree node
        :param y: hypothesis tree node
        :param deltaed: threshold
        :return: distance
        """
        if (x is None or y is not None) or (x is not None and y is None): # from the tree-edit definition
            return 1.0
        else:
            v1 = self.vec.get_vector(x).reshape(1,-1)
            v2 = self.vec.get_vector(y).reshape(1,-1)
            dist = cosine_similarity(v1,v2)[0][0]

            if dist < deltaed:
                return 1.0
            else:
                return round(dist, 4)

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

    def semantic_ted(self,ref,hyp):
        """
        Computes semantic tree edit distance
        """
        y_nodes, y_adj = self.stanza2ted(text=self.clean_str(ref), key="lemma")

        if len(hyp) == 0:
            x_nodes, x_adj = [''], [[]]
        else:
            x_nodes, x_adj = self.stanza2ted(text=self.clean_str(hyp), key="token")

        tedval = ted.ted(x_nodes, x_adj, y_nodes, y_adj, delta=self.diff)
        sc = max(0,(tedval-1.0))
        return round(sc/(len(y_nodes)+1e-30),4)


if __name__=="__main__":
    tree = TEDse()

    hyp = "103 hera discoverer james craig watson james craig watson deathcause peritonitis"
    ref = "james craig watson , who died of peritonitis , was the discoverer of 103 hera ."
    print("Semantic-TED: ", tree.semantic_ted(ref, hyp))
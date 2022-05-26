from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class Grammar:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("saved_model/grammar/", use_auth_token=False)
        self.model = AutoModelForSequenceClassification.from_pretrained("saved_model/grammar/")

    def check_grammar(self, sent):
        """
        :param sent: takes a sentence as input
        :return: grammatical acceptability score
        """
        inputs = self.tokenizer(sent, return_tensors="pt")
        classification_logits = self.model(**inputs)[0]
        pred_labels = torch.softmax(classification_logits, dim=1)

        pred_class = torch.argmax(pred_labels).item()  # class label 0 or 1;  0 -> non-grammatical, 1 -> grammatical
        pred_class_score = torch.max(pred_labels).item()  # class score

        return round(pred_class_score,4) if pred_class else round(1 - pred_class_score, 4)


if __name__ == "__main__":

    g = Grammar()
    print(g.check_grammar("Not want she go shop sunny"))
    print(g.check_grammar("Could you please pass me the salt?"))
    print(g.check_grammar("Elon elon elon mask is the founder."))

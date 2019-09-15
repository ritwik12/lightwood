import torch
from pytorch_transformers import DistilBertModel, DistilBertTokenizer


class BertEncoder:
    def __init__(self):
        self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self._model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self._pytorch_wrapper = torch.FloatTensor

    def encode(self, column_data):

        tokenized_inputs = []
        for text in column_data:
            tokenized_input = torch.tensor([self._tokenizer.encode(text, add_special_tokens=True)])
            tokenized_inputs.append(tokenized_input)

        encoded_representation = []
        with torch.no_grad():
            for tokenized_input in tokenized_inputs:
                last_hidden_states = self._model(tokenized_input)[0]
                encoded_representation.append(last_hidden_states)
                print(last_hidden_states)
                exit()
        return self._pytorch_wrapper(encoded_representation)


    def decode(self, encoded_values_tensor, max_length = 100):
        # When test is an output... a bit trickier to handle this case, thinking on it
        pass

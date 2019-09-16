import torch
from pytorch_transformers import DistilBertModel, DistilBertTokenizer


class BertEncoder:
    def __init__(self):
        self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self._model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self._pytorch_wrapper = torch.FloatTensor

    def encode(self, column_data):
        encoded_representation = []

        tokenized_inputs = []
        for text in column_data:
            tokenized_input = torch.tensor([self._tokenizer.encode(text, add_special_tokens=True)])

            encoded_representation.append(list(tokenized_input[0]))

            tokenized_input = torch.tensor(list(tokenized_input[0][0:512]))
            tokenized_inputs.append(tokenized_input)

        max_len = max([len(x) for x in encoded_representation])
        max_element = 0
        for arr in encoded_representation:
            for ele in arr:
                if ele > max_element:
                    max_element = ele

        for arr in encoded_representation:
            while len(arr) < max_len:
                arr.append(0)
            for i in range(len(arr)):
                arr[i] = arr[i]/max_element

        return self._pytorch_wrapper(encoded_representation)

        # See this later
        with torch.no_grad():
            for tokenized_input in tokenized_inputs:
                last_hidden_states = self._model(tokenized_input)[0]
                encoded_representation.append(last_hidden_states)


    def decode(self, encoded_values_tensor, max_length = 100):
        # When test is an output... a bit trickier to handle this case, thinking on it
        pass

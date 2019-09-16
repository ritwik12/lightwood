import torch
from pytorch_transformers import DistilBertModel, DistilBertTokenizer


class BertEncoder:
    def __init__(self):
        self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self._model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self._pytorch_wrapper = torch.FloatTensor
        self._max_len = None
        self._max_ele = None

    def encode(self, column_data):
        encoded_representation = []
        print(self._max_len)
        tokenized_inputs = []
        for text in column_data:
            tokenized_input = torch.tensor([self._tokenizer.encode(text, add_special_tokens=True)])

            encoded_representation.append(list(tokenized_input[0]))

            tokenized_input = torch.tensor(list(tokenized_input[0][0:512]))
            tokenized_inputs.append(tokenized_input)

        if self._max_len is None:
            self._max_len = max([len(x) for x in encoded_representation])

        if self._max_ele is None:
            self._max_ele = 0
            for arr in encoded_representation:
                for ele in arr:
                    if ele > self._max_ele:
                        self._max_ele = ele

        for arr in encoded_representation:
            while len(arr) < self._max_len:
                arr.append(0)
            while len(arr) > self._max_len:
                arr.pop(self._max_len())
                
            for i in range(len(arr)):
                arr[i] = arr[i]/self._max_ele

        return self._pytorch_wrapper(encoded_representation)

        # See this later
        with torch.no_grad():
            for tokenized_input in tokenized_inputs:
                last_hidden_states = self._model(tokenized_input)[0]
                encoded_representation.append(last_hidden_states)


    def decode(self, encoded_values_tensor, max_length = 100):
        # When test is an output... a bit trickier to handle this case, thinking on it
        pass

from pytorch_transformers import RobertaModel, RobertaTokenizer


class BertEncoder:
    def __init__(self):
        self.tokenizer = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print(self.model)
        exit()

    def encode(self, column_data):

        tokenized_inputs = []
        for text in column_data:
            tokenized_input = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
            tokenized_inputs.append(tokenized_input)

        encoded_representation = []
        with torch.no_grad():
            for tokenized_input in tokenized_inputs:
                last_hidden_states = model(tokenized_input)[0]
                print(last_hidden_states)
                encoded_representation.append(last_hidden_states)

        return self._pytorch_wrapper(encoded_representation)


    def decode(self, encoded_values_tensor, max_length = 100):
        # When test is an output... a bit trickier to handle this case, thinking on it
        pass

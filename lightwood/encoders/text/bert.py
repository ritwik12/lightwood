from pytorch_transformers import RobertaModel, RobertaTokenizer


class BertEncoder:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')

    def encode(self, column_data):

        print(column_data)
        exit()

        return self._pytorch_wrapper(ret)


    def decode(self, encoded_values_tensor, max_length = 100):

        ret = []
        with torch.no_grad():
            for decoder_hiddens in encoded_values_tensor:
                decoder_hidden = torch.FloatTensor([[decoder_hiddens.tolist()]])


                decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

                decoded_words = []

                for di in range(max_length):
                    decoder_output, decoder_hidden = self._decoder(
                        decoder_input, decoder_hidden)

                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(self._output_lang.index2word[topi.item()])

                    decoder_input = topi.squeeze().detach()

                ret += [' '.join(decoded_words)]

        return ret

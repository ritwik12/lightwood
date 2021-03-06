import torch
from PIL import Image
import requests
from io import BytesIO

from lightwood.encoders.image.helpers.img_to_vec import Img2Vec
from lightwood.config.config import CONFIG
from lightwood.constants.lightwood import ENCODER_AIM

class Img2VecEncoder:

    def __init__(self, is_target=False, aim=ENCODER_AIM.BALANCE):
        self._model = None
        # I think we should make this an enum, something like: speed, balance, accuracy
        self.aim = aim
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False

    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        if self._model is None:
            if self.aim == ENCODER_AIM.SPEED:
                self._model = Img2Vec(model='resnet-18')
            elif self.aim == ENCODER_AIM.BALANCE:
                self._model = Img2Vec(model='resnext-50-small')
            elif self.aim == ENCODER_AIM.ACCURACY:
                self._model = Img2Vec(model='resnext-50')
            else:
                self._model = Img2Vec()
        self._prepared = True

    def encode(self, images):
        """
            Encode list of images

            :images : list of images, each image is a path image(ToDO: url to image also need to be included)
            :return: a torch.floatTensor
        """
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        pics = []
        for image in images:
            if image.startswith('http'):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image)

            vec = self._model.get_vec(img)
            pics.append(vec)

        return torch.FloatTensor(pics)


if __name__ == "__main__":
    images = ['test_data/cat.jpg', 'test_data/cat2.jpg', 'test_data/catdog.jpg']

    encoder = Img2VecEncoder()

    ret = encoder.encode(images)
    print(ret)

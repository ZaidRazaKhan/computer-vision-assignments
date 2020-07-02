import sys
import os
from deep_features import Img2Vec
from PIL import Image

img2vec = Img2Vec(cuda = True, model = 'alexnet', layer = 'default', layer_output_size = 4096)


img = Image.open('./image_2.png')
vec = img2vec.get_vec(img)
print(vec)


import streamlit as st
import numpy as np
import torchvision
import time

import torch

from models_arc import *

n_classes = 10
classes = ['T-shirt/top',
           'Trouser',
           'Pullover',
           'Dress',
           'Coat',
           'Sandal',
           'Shirt',
           'Sneaker',
           'Bag',
           'Ankle boot']

model = ConvModel_3(n_classes)
model.eval()
model.load_state_dict(torch.load('final_BatchNorm.pt'))
test_dataset = torchvision.datasets.FashionMNIST('data',train=False, download=True)

def load_image(image_file):
  img = Image.open(image_file)
  return img


st.title("Fashion MNIST Demo App!")
if st.button('Pick random image from test'):
    n = np.random.randint(0, len(test_dataset))
    img_tensor = test_dataset.data[n] / 255.0
    img = img_tensor.numpy()
    st.image(img,  use_column_width=True)

    prediction = model(img_tensor.view(1,1,28,28)).detach().view(-1).numpy()
    probs = np.exp(prediction)
    ans = classes[np.argmax(probs)]
    p = np.round(probs[np.argmax(probs)],3)
    print(p)
    st.success(f"It's a {ans} with probability {p}")

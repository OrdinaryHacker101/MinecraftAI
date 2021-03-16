import os

from dynamics_engine import ActionLSTM

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

path = os.path.join("C:\\Users\\Robin\\Desktop\\deep_learning")
file = np.load(path+"\\processed_data\\v3_absolute_grape_changeling-12_2293-4124NDTRAIN.npy", allow_pickle=True)[1]

file = np.array(file[-1792:][:1700]).astype(np.uint8)

processed = file.reshape(1700, 64, 64, 3)
test = processed[-1]
test = test.reshape(-1,64,64,3)
processed = processed[:-1]
test = test.reshape(-1,64,64,3).astype(np.float32)/255.
processed = processed.astype(np.float32)/255.
print(processed[0].shape)
plt.imshow(processed[-1])
plt.show()
print(processed[0])

test_o = processed[0]
test_t = processed[100]

h_and_c = ActionLSTM(input_dim=64).forward(h=np.array([[test_t]]), c=np.array([[10]]), INPUT=np.array([test_o]), state_bias=None, step=0)

print("tensors: ", h_and_c)

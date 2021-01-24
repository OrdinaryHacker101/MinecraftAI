import minerl
import gym

import tensorflow as tf
import numpy as np
import time

import matplotlib.pyplot as plt
import tqdm

action_labels = ["attack", "back", "camera", "forward", "jump", "left", "right", "sneak", "sprint"]
obs_metalabels = ["compassAngle", "inventory", "pov"]
model = tf.keras.models.load_model("C:\\Users\\Robin\\Desktop\\deep_learning\\models\\AZNET.h5")

def main():
    reward = 0
    env = gym.make("MineRLNavigateDense-v0")
    obs = env.reset()
    done = False
    while not done:
        img = env.render(mode="rgb_array")
        obs_labels = [obs[j] for j in obs_metalabels[:-1]]
        print("Labels: ",obs_labels)
        obs_labels[-1] = obs_labels[-1]["dirt"]
        obs_img = obs[obs_metalabels[-1]]
        obs_labels.insert(0, reward)
        obs_labels = np.asarray(obs_labels).astype(np.float32)
        print(obs_labels)
        print(img.shape)
        predictions = model.predict([np.array(img).reshape(1, 64, 64, 3), np.array(obs_labels).reshape(1,3)])
        print("Predictions: ", predictions)
        obs, reward, done, info = env.step({"forward": 1})
        #print("Observations with no image: ", obs_labels)

if __name__ == "__main__":
    main()

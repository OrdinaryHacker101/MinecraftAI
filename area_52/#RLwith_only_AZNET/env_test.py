import minerl
import gym

import tensorflow as tf
import numpy as np
import time

import matplotlib.pyplot as plt
import tqdm

action_labels = ["forward", "left", "back", "right", "jump", "sneak", "sprint", "attack", "place", "camera"]
obs_metalabels = ["compassAngle", "inventory", "pov"]

#AZNET
model = tf.keras.models.load_model("C:\\Users\\Robin\\Desktop\\deep_learning\\models\\AZNET.h5")

def main():
    for i in range(10):
        reward = 0

        #the environment for the task of navigation
        env = gym.make("MineRLNavigateDense-v0")
        obs = env.reset()
        done = False
        steps = 0

        memory = np.array([])
    
        for steps in range(200):
            img = env.render(mode="rgb_array")
            
            #random variables
            rand_action_choice = np.random.randint(8)
            
            #putting observation data in a list
            obs_labels = [obs[j] for j in obs_metalabels[:-1]]
            print("Labels: ",obs_labels)
            obs_labels[-1] = obs_labels[-1]["dirt"]
            obs_img = obs[obs_metalabels[-1]]
            obs_labels.insert(0, reward)
            obs_labels = np.asarray(obs_labels).astype(np.float32)
            print(obs_labels)
            print(img.shape)

            #calculated actions 
            predictions = model.predict([np.array(img).reshape(1, 64, 64, 3), np.array(obs_labels).reshape(1,3)])
            predictions = np.rint(predictions).astype(np.int32)
            #camera actions calculated my the model
            camera_x = predictions[0][-2]
            camera_y = predictions[0][-1]
            predictions = predictions[0][:-2]
            
            print("Predictions: ", predictions)

            #if the reward is less than 1, which means the action the neural network produced is HORRIBLE, tell it to modify the predictions
            if reward < 1:
                predictions[rand_action_choice] += np.sqrt(reward)
            
            #This is the code for doing the action in the environment
            for i in range(len(predictions)):
                n_obs, n_reward, n_done, n_info = env.step({action_labels[int(i)]: int(predictions[int(i)])})

            #code for moving the camera
            obs, reward, done, info = env.step({"camera": [camera_x, camera_y]})

            #np.append(memory, [n_reward + reward, int(predictions), img])
            steps += 1
        

if __name__ == "__main__":
    main()

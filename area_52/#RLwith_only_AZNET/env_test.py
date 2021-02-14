import minerl
import gym

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

import matplotlib.pyplot as plt
import tqdm

action_labels = ["forward", "left", "back", "right", "jump", "sneak", "sprint", "attack", "place", "camera"]
obs_metalabels = ["compassAngle", "inventory", "pov"]

#AZNET
model = tf.keras.models.load_model("C:\\Users\\Robin\\Desktop\\deep_learning\\models\\AZNET.h5")

#training hyperparameters
ITERATIONS = 100
EPISODES = 10
MAX_STEPS = 100
DISCOUNT_FACTOR = 0.6

OPTIMIZER = keras.optimizers.Adam(lr=0.01)
LOSS_FN = keras.losses.binary_crossentropy
current_episode = 0
basic_size = 4
rewards = 0

def use_obs_in_model(obs, reward):
    
    #putting observation data in a list
    obs_labels = [obs[j] for j in obs_metalabels[:-1]]
    print("Labels: ",obs_labels)
    
    obs_labels[-1] = obs_labels[-1]["dirt"]
    obs_img = obs[obs_metalabels[-1]]
    obs_labels.insert(0, reward)
    obs_labels = np.asarray(obs_labels).astype(np.float32)
    print(obs_labels)
    return obs_labels

def play_one_step(env, obs, img, model, loss_fn, reward):
    with tf.GradientTape() as tape:
        preds = model([np.array(img).reshape(1, 64, 64, 3), use_obs_in_model(obs, reward).reshape(1,3)])
        
        camera_x = preds[0][-2]
        camera_y = preds[0][-1]

        actions = tf.cast((tf.random.uniform([1, 9]) > preds[0][:-2]), tf.float32)
        y_target = tf.ones([1, 9], tf.float32) - actions
        
        print("Actions: ", actions)
        loss = tf.reduce_mean(loss_fn(y_target, actions))

    grads = tape.gradient(loss, model.trainable_variables)
    #This is the code for doing the action in the environment
    for i in range(len(actions[0])):
        n_obs, n_reward, n_done, n_info = env.step({action_labels[int(i)]: int(actions[0][int(i)])})
    
    #code for moving the camera
    obs, reward, done, info = env.step({"camera": [camera_x*180, camera_y*180]})
    rewards = reward
    print(rewards)
    return obs, reward, done, grads

def play_multiple_episodes(env, EPISODES, MAX_STEPS, img, model, loss_fn, reward):
    all_rewards = []
    all_grads = []
    for episode in range(EPISODES):
        print("EPISODE: ",episode)
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(MAX_STEPS):
            obs, REWARD, done, grads = play_one_step(env, obs, img, model, loss_fn, reward)
            current_rewards.append(REWARD)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads

def discount_rewards(reward, discount_factor):
    discounted = np.array(reward)
    for step in range(len(reward) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(reward, discount_factor)
                              for reward in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]

def train_model():
    env = gym.make("MineRLNavigateDense-v0")
    obs = env.reset()
    done = False
    for iteration in range(ITERATIONS):
        img = env.render(mode="rgb_array")

        all_rewards, all_grads = play_multiple_episodes(
            env, EPISODES, MAX_STEPS, img, model, LOSS_FN, rewards)
        all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                           discount_factor)
        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                 for episode_index, final_rewards in enumerate(all_final_rewards)
                     for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        OPTIMIZER.apply_gradients(zip(all_mean_grads, model.trainable_variables))
    
def main():

    #the environment for the task of navigation
    env = gym.make("MineRLNavigateDense-v0")
    obs = env.reset()
    done = False
    steps = 0

    while not done:
        img = env.render(mode="rgb_array")
        print(img.shape)

        obs_labels = use_obs_in_model(obs)
        
        #calculated actions 
        predictions = model.predict([np.array(img).reshape(1, 64, 64, 3), np.array(obs_labels).reshape(1,3)])
        predictions = np.rint(predictions).astype(np.int32)
        #camera actions calculated my the model
        camera_x = predictions[0][-2]
        camera_y = predictions[0][-1]
        predictions = predictions[0][:-2]
        
        print("Predictions: ", predictions)
        
        #This is the code for doing the action in the environment
        for i in range(len(predictions)):
            n_obs, n_reward, n_done, n_info = env.step({action_labels[int(i)]: int(predictions[int(i)])})

        #code for moving the camera
        obs, reward, done, info = env.step({"camera": [camera_x*180, camera_y*180]})
        print(info)


if __name__ == "__main__":
    train_model()
    #main()

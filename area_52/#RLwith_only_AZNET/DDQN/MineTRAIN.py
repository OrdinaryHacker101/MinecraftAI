import minerl
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

from MineDDDQNPG import Agent

action_labels = ["forward", "left", "back", "right", "jump", "sneak", "sprint", "attack", "place", "camera"]
obs_metalabels = ["compassAngle", "inventory", "pov"]

model = tf.keras.models.load_model("C:\\Users\\Robin\\Desktop\\deep_learning\\models\\AZNET.h5")

current_episode = 0
basic_size = 4
rewards = 0

done = False

def use_obs_in_model(obs, reward):
    
    #putting observation data in a list
    obs_labels = [obs[j] for j in obs_metalabels[:-1]]
    print("Labels: ",obs_labels)
    
    obs_labels[-1] = obs_labels[-1]["dirt"]
    obs_img = obs[obs_metalabels[-1]]
    obs_labels.insert(0, reward)
    obs_labels = np.asarray(obs_labels).astype(np.float32)
    print(obs_img)
    print(obs_img.shape)
    print(np.array(obs_img).dtype)
    print(obs_labels)
    print(np.array(obs_labels).dtype)
    return [np.array(obs_img).reshape(1, 64, 64, 3).astype(np.float32), obs_labels.reshape(1,3)]

def train_model():
    
    env = gym.make("MineRLNavigateDense-v0")
    
    agent = Agent(lr=0.0005, gamma=0.99, n_actions=9, main_model=model, epsilon=1.0,
                  batch_size=64, input_dims=[2])
    n_games = 500
    scores = []
    for i in range(n_games):
        done = False
        score = 0
        reward = 0
        obs = env.reset()
        print(obs)
        while not done:
            img = env.render(mode="rgb_array")
            action = agent.choose_action(use_obs_in_model(obs, reward))
            print(action)
            
            obs_, reward, done, info = env.step({action_labels[action]: 1})
            score += reward
            camera_x, camera_y = agent.mouse_control(use_obs_in_model(obs_, reward))
            
            obs_, reward, done, info = env.step({"camera": [camera_x*180, camera_y*180]})
            score += reward
            state = use_obs_in_model(obs, reward)
            state_ = use_obs_in_model(obs_, reward)
            agent.store_transition(state[0], state[1], action, reward, state_[0], state_[1], done)
            obs = obs_
            agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("episode ", i, "avg score %.1f" % avg_score)
        
        
if __name__ == "__main__":
    train_model()

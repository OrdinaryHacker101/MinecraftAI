import minerl
import gym

import tensorflow as tf
import numpy as np
import cv2

from collections import deque
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import tqdm
import os

#from time import time

#minerl.data.download("C:\\Users\\Robin\\Desktop\\deep_learning", experiment="MineRLNavigateDense-v0")
#ACTIONS = ["attack", "back", "camera", "forward", "jump", "left", "right", "sneak", "sprint"]

data_params = ['reward', 'observation$compassAngle', 'observation$inventory$dirt', 'action$forward', 'action$left', 'action$back', 'action$right', 'action$jump', 'action$sneak', 'action$sprint', 'action$attack', 'action$camera', 'action$place']

basic_actions = ["back", "forward", "jump", "left", "right", "camera"]
data_rewards = []
data_states = []

#total_training_data =

training_data_path = "C:\\Users\\Robin\\Desktop\\deep_learning\\MineRLNavigateDense-v0"
np_data_file = "rendered.npz"
video_file = "recording.mp4"

replay_buffer = deque(maxlen=2000)

batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
loss_fn = tf.keras.losses.mean_squared_error

def make_data():
    
    
    for folder in tqdm.tqdm(os.listdir(training_data_path)):
        images = np.array([])
        path = os.path.join(training_data_path+"\\"+folder)
        data = np.load(path+"\\"+np_data_file)
        video = cv2.VideoCapture(path+"\\"+video_file)
        success, image = video.read()

        while success:
            try:
                success, image = video.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #plt.imshow(image)
                #plt.show()
                images = np.append(images, image)
                print(images)
            except:
                continue
        
        training_data = np.array([images, [data[i] for i in data_params]])
        #total_training_data = np.append(total_training_data, training_data)
        
        np.save(os.path.join(path,"NDTRAIN.npy"), training_data)

def preprocess_data():
    training_data = np.load("NDTRAIN.npy", allow_pickle=True)
    to_be_clustered = []
    print(training_data[0])
     
    for image in training_data[0]:
        print("The loop is running")
        print(image)
        plt.imshow(image)
        plt.show()

def build_visual_model():
    print("Running the convolutional neural network")
    v_model = tf.keras.models.Sequential([
    
        tf.keras.layers.Conv2D(32, 5, activation="relu", padding="same", input_shape=[64, 64, 3]),
        tf.keras.layers.MaxPooling2D((2,2), padding="same"),
        tf.keras.layers.Conv2D(64, 5, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2,2), padding="same"),
        tf.keras.layers.Conv2D(64, 5, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2,2), padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="softmax")

    ])
    
    v_model.summary()

    return v_model
    

def build_action_model():

    env = gym.make("MineRLNavigateDense-v0")

    v_model = build_visual_model()
    
    d_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(60, activation="relu", input_shape = [3]),
        tf.keras.layers.Dense(60, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax")
    ])
    d_model.summary()

    def epsilon_greedy_policy(state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(5)
        else:

            print("The artificial brain is in use")
            '''state_obs = list(state.items())
            state_obs[1][1] = 0'''
            state_img = state["pov"]
            state_img = state_img.reshape(-1, 64, 64, 3)
            '''plt.imshow(state_img)
            plt.show()'''
            
            img_preds = v_model.predict(state_img)
            
            state_obs = [state["compassAngle"], state["inventory"]["dirt"], img_preds[-1].astype(np.float32)]

            Q_values = d_model.predict(np.asarray(state_obs).astype(np.float32)[np.newaxis])
            
            
            print("Q values: ",Q_values)
            return np.argmax(Q_values[0])

    def sample_experiences(batch_size):
        indices = np.random.randint(len(replay_buffer), size=batch_size)
        batch = [replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def play_one_step(env, state, epsilon):
        action = epsilon_greedy_policy(state, epsilon)
        print(action)
        next_state, reward, done, info = env.step({basic_actions[action]: 1})
        replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done, info
    
    def training_step(batch_size):
        experiences = sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = d_model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values)
        target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)
        mask = tf.one_hot(actions, n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.appy_gradients(zip(grads, model.trainable_variables))

    episode_rewards = []
    
    for episode in tqdm.tqdm(range(600)):
        
        episode_rewards = sum(episode_rewards)
        print("Episode rewards: ",episode_rewards)
        episode_rewards = []
        
        env.seed(420)
        obs = env.reset()
        epsilon = max(1 - episode / 500, 0.01)
        for step in range(200):
            epsilon = max(1 - episode / 500, 0.01)
            obs, reward, done, info = play_one_step(env, obs, epsilon)
            print("Rewards: ", reward)
            episode_rewards.append(reward)
            if done:
                break

        if episode > 50:
            training_step(batch_size)

def build_memory_model():
    pass

def build_decision_model():
    action_d = build_action_model()
    decision_model = tf.keras.models.Sequential()

def train_model():
    pass
        
def look_and_make_data():
    n_data = minerl.data.make(
    "MineRLNavigateDense-v0",
    data_dir="C:\\Users\\Robin\\Desktop\\deep_learning")
    print(n_data)

    to_be_clustered = []
    
    for current_state, action, reward, next_state, done \
        in tqdm.tqdm(n_data.batch_iter(16, 32, 2, preload_buffer_size=20)):
        '''plt.imshow(np.array(current_state["pov"][-1][-1]).reshape((64,64,3)))
        plt.show()'''

        print(action)

        '''to_be_clustered.append(action["vector"])
        if len(to_be_clustered) > 1000:
            break

    actions = np.concatenate(to_be_clustered).reshape(-1, 64)
    kmeans_actions = actions[:100000]
    kmeans = KMeans(n_clusters = 32, random_state=0).fit(kmeans_actions)
    print(kmeans)'''

        
        
    

def main():
    env = gym.make("MineRLNavigateDense-v0")
    obs = env.reset()
    done = False
    while not done:
        img = env.render(mode="rgb_array")
        obs, reward, done, info = env.step({"forward": 1})
        print("Observations: ",obs)
    

if __name__ == "__main__":
    make_data()
    #preprocess_data()
    #build_action_model()
    #main()

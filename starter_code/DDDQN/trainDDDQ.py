import gym
import numpy as np
from DDDQ import Agent

if __name__ == "__main__":
    #creates the lunar lander environment.
    env = gym.make("LunarLander-v2")

    #calls the agent
    agent = Agent(lr=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
                  batch_size=64, input_dims=[8])

    #defines the number of games to be played, the scores, and the epsilons that
    #control whether the action is calculated by the models or is random.
    n_games = 500
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            #moves the action
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        #calculates the average score
        avg_score = np.mean(scores[-100:])
        print("episode ", i, "avg score %.1f" % avg_score)

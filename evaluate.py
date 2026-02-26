import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import DQN
import argparse

gym.register_envs(ale_py)

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)

args = parser.parse_args()

env = gym.make("ALE/Pacman-v5")

model = DQN.load(args.model)

episodes = 20
scores = []

for ep in range(episodes):

    obs, _ = env.reset()
    done = False
    total = 0

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        total += reward

    scores.append(total)

print("Mean:", np.mean(scores))
print("Std:", np.std(scores))
print("Max:", np.max(scores))
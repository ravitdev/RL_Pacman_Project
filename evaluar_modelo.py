import argparse
import numpy as np
import gymnasium as gym
import ale_py
from train import CustomDQN


parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ddqn"])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--episodes", type=int, default=10)

args = parser.parse_args()

ALGO = args.algo.lower()
SEED = args.seed
EPISODES = args.episodes

gym.register_envs(ale_py)

env = gym.make("ALE/Pacman-v5")

model_path = f"models/{ALGO}_seed_{SEED}/final_model.zip"

print(f"Cargando modelo desde: {model_path}")

model = CustomDQN.load(model_path, env=env)

scores = []

for episode in range(1, EPISODES + 1):
    obs, info = env.reset(seed=SEED + episode)
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated

    scores.append(score)
    print(f"Episodio {episode} - Score: {score}")

mean_score = np.mean(scores)
std_score = np.std(scores)

print("\n" + "="*50)
print(f"Algoritmo: {ALGO.upper()} | Seed: {SEED}")
print(f"Episodios evaluados: {EPISODES}")
print(f"Recompensa media: {mean_score:.2f}")
print(f"Desviación estándar: {std_score:.2f}")
print(f"Scores individuales: {scores}")
print("="*50)

env.close()
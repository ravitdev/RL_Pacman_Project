import gymnasium as gym
import ale_py
import time

# Registrar entornos Atari
gym.register_envs(ale_py)

ENV_NAME = "ALE/Pacman-v5"

# Crear entorno con render
env = gym.make(ENV_NAME, render_mode="human")

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # acción aleatoria
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(0.02)  # velocidad para que puedas verlo

env.close()
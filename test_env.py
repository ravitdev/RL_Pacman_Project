import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Pacman-v5", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
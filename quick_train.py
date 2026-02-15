import gymnasium as gym
import ale_py
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Registrar Atari
gym.register_envs(ale_py)

def make_env():
    return gym.make("ALE/Pacman-v5")

env = DummyVecEnv([make_env])

print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    train_freq=4,
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda"
)

model.learn(total_timesteps=50000)
model.save("models/test_model")

env.close()
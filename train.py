import os
import time
import argparse
import numpy as np
import gymnasium as gym
import ale_py
import torch
import torch.nn.functional as F

from stable_baselines3.dqn.dqn import DQN as SB3_DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

gym.register_envs(ale_py)

# =========================
# ARGUMENTOS
# =========================

parser = argparse.ArgumentParser()

parser.add_argument("--algo", type=str, default="dqn")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--timesteps", type=int, default=200000)

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--buffer_size", type=int, default=50000)
parser.add_argument("--train_freq", type=int, default=4)

parser.add_argument("--loss", type=str, default="huber")

args = parser.parse_args()

# =========================
# CONFIG
# =========================

ALGO = args.algo.lower()
SEED = args.seed
TIMESTEPS = args.timesteps

LR = args.lr
BATCH = args.batch_size
BUFFER = args.buffer_size
TRAIN_FREQ = args.train_freq
LOSS = args.loss.lower()

ENV_NAME = "ALE/Pacman-v5"

exp_name = f"final_{ALGO}_{TIMESTEPS}_lr{LR}_bs{BATCH}_buf{BUFFER}_tf{TRAIN_FREQ}_{LOSS}_seed{SEED}"

log_dir = f"logs/{exp_name}"
model_dir = f"models/{exp_name}"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# =========================
# SEEDS
# =========================

set_random_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# CUSTOM DQN / DDQN
# =========================

class CustomDQN(SB3_DQN):

    def __init__(self, *args, loss_type="huber", double_q=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.double_q = double_q

    def train(self, gradient_steps: int, batch_size: int = 100):

        for _ in range(gradient_steps):

            replay_data = self.replay_buffer.sample(batch_size)

            with torch.no_grad():

                if self.double_q:
                    # Double DQN
                    next_actions = self.q_net(
                        replay_data.next_observations
                    ).argmax(dim=1)

                    next_q_target = self.q_net_target(
                        replay_data.next_observations
                    )

                    next_q = next_q_target.gather(
                        1,
                        next_actions.unsqueeze(1)
                    ).squeeze(1)

                else:
                    # Standard DQN
                    next_q_target = self.q_net_target(
                        replay_data.next_observations
                    )

                    next_q, _ = next_q_target.max(dim=1)

                target_q = replay_data.rewards.flatten() + (
                    1 - replay_data.dones.flatten()
                ) * self.gamma * next_q

            current_q = self.q_net(replay_data.observations)

            current_q = current_q.gather(
                1,
                replay_data.actions.long()
            ).flatten()

            if self.loss_type == "huber":
                loss = F.smooth_l1_loss(current_q, target_q)
            else:
                loss = F.mse_loss(current_q, target_q)

            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

# =========================
# ENV
# =========================

def make_env():
    env = gym.make(ENV_NAME)
    env = Monitor(env, log_dir)
    env.reset(seed=SEED)
    return env

env = DummyVecEnv([make_env])

# =========================
# MODELO
# =========================

model = CustomDQN(
    "CnnPolicy",
    env,
    learning_rate=LR,
    batch_size=BATCH,
    buffer_size=BUFFER,
    learning_starts=10000,
    gamma=0.99,
    train_freq=TRAIN_FREQ,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    tensorboard_log=log_dir,
    verbose=1,
    seed=SEED,
    device="cuda",
    loss_type=LOSS,
    double_q=(ALGO == "ddqn")
)

# =========================
# ENTRENAMIENTO
# =========================

start = time.time()

model.learn(total_timesteps=TIMESTEPS)

training_time = time.time() - start

# =========================
# GUARDAR
# =========================

model.save(f"{model_dir}/model")

with open(f"{log_dir}/time.txt", "w") as f:
    f.write(str(training_time))

print("\nTraining finished")
print("Algorithm:", ALGO.upper())
print("Seed:", SEED)
print("Timesteps:", TIMESTEPS)
print("Training time (seconds):", training_time)
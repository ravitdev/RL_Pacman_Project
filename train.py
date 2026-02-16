import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import ale_py

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


# =========================================================
# ARGUMENTOS
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ddqn"])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--timesteps", type=int, default=1_000_000)

args = parser.parse_args()

ALGO = args.algo.lower()
SEED = args.seed
TIMESTEPS = args.timesteps


# =========================================================
# CARPETAS (definidas antes del entorno)
# =========================================================

log_dir = f"logs/{ALGO}_seed_{SEED}"
model_dir = f"models/{ALGO}_seed_{SEED}"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# =========================================================
# CONFIGURACIÓN REPRODUCIBLE
# =========================================================

set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("=" * 60)
print(f"Algoritmo      : {ALGO}")
print(f"Seed           : {SEED}")
print(f"Timesteps      : {TIMESTEPS}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
print("=" * 60)


# =========================================================
# CLASE CUSTOM DQN (permite activar/desactivar Double Q)
# =========================================================

class CustomDQN(DQN):
    """
    double_q = False  -> DQN clásico
    double_q = True   -> Double DQN
    """

    def __init__(self, *args, double_q: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.double_q = double_q

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        for _ in range(gradient_steps):

            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with torch.no_grad():

                if self.double_q:
                    # ==============================
                    # DOUBLE DQN
                    # ==============================
                    next_actions = self.q_net(
                        replay_data.next_observations
                    ).argmax(dim=1)

                    next_q_values = self.q_net_target(
                        replay_data.next_observations
                    )

                    next_q_values = next_q_values.gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze(1)

                else:
                    # ==============================
                    # DQN CLÁSICO
                    # ==============================
                    next_q_values = self.q_net_target(
                        replay_data.next_observations
                    )
                    next_q_values, _ = next_q_values.max(dim=1)

                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones
                ) * self.gamma * next_q_values

            current_q_values = self.q_net(
                replay_data.observations
            )

            current_q_values = current_q_values.gather(
                1, replay_data.actions.long()
            ).squeeze(1)

            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm
            )
            self.policy.optimizer.step()

        self._n_updates += gradient_steps


# =========================================================
# REGISTRAR ATARI
# =========================================================

gym.register_envs(ale_py)


def make_env():
    env = gym.make("ALE/Pacman-v5")
    env = Monitor(env, filename=f"{log_dir}/monitor.csv")
    env.reset(seed=SEED)
    return env


env = DummyVecEnv([make_env])


# =========================================================
# CREAR MODELO
# =========================================================

double_q_flag = True if ALGO == "ddqn" else False

model = CustomDQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    train_freq=4,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log=log_dir,
    seed=SEED,
    device="cuda",
    double_q=double_q_flag,
)


# =========================================================
# ENTRENAMIENTO
# =========================================================

start_time = time.time()

model.learn(total_timesteps=TIMESTEPS)

end_time = time.time()
training_time = end_time - start_time

print("=" * 60)
print(f"Tiempo total de entrenamiento: {training_time/60:.2f} minutos")
print("=" * 60)


# =========================================================
# GUARDAR MODELO
# =========================================================

model.save(f"{model_dir}/final_model")
env.close()
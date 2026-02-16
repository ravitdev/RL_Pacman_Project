import argparse
import os
import numpy as np
import gymnasium as gym
import ale_py
import torch
from stable_baselines3.dqn.dqn import DQN

# =========================================================
# CLASE CUSTOM DQN
# Redefinimos la clase aquí para que el script sea independiente
# y no intente ejecutar el código de entrenamiento de train.py
# =========================================================
class CustomDQN(DQN):
    def __init__(self, *args, double_q: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.double_q = double_q

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # No necesitamos la lógica de entrenamiento para evaluar
        pass

# =========================================================
# CONFIGURACIÓN DE ARGUMENTOS
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de modelos DQN/DDQN")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ddqn"], help="Algoritmo a evaluar")
    parser.add_argument("--seed", type=int, default=1, help="Seed del modelo guardado")
    parser.add_argument("--episodes", type=int, default=10, help="Número de episodios a evaluar")
    parser.add_argument("--render", action="store_true", help="Activar para ver la pantalla del juego")

    args = parser.parse_args()

    ALGO = args.algo.lower()
    SEED = args.seed
    EPISODES = args.episodes
    
    # Decidir modo de renderizado
    render_mode = "human" if args.render else None

    # =========================================================
    # ENTORNO Y CARGA DE MODELO
    # =========================================================
    gym.register_envs(ale_py)
    
    try:
        env = gym.make("ALE/Pacman-v5", render_mode=render_mode)
    except Exception as e:
        print(f"Error al crear el entorno: {e}")
        # Fallback por si el render_mode falla en algún sistema
        env = gym.make("ALE/Pacman-v5")

    model_path = f"models/{ALGO}_seed_{SEED}/final_model.zip"

    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: No se encontró el modelo en: {model_path}")
        print(f"Asegúrate de que la carpeta 'models' esté en: {os.getcwd()}")
        env.close()
        exit()

    print(f"\n" + "="*50)
    print(f"Cargando modelo: {ALGO.upper()} | SEED: {SEED}")
    print(f"Modo Visual: {'ACTIVADO' if args.render else 'DESACTIVADO'}")
    print("="*50)

    # Cargamos el modelo (forzamos CPU para evaluar sin ocupar la GPU innecesariamente)
    model = CustomDQN.load(model_path, env=env, device="cpu")

    scores = []

    # =========================================================
    # BUCLE DE EVALUACIÓN
    # =========================================================
    for episode in range(1, EPISODES + 1):
        # Usamos un seed diferente por episodio para probar robustez
        obs, info = env.reset(seed=SEED + episode + 100)
        done = False
        score = 0

        while not done:
            # deterministic=True usa la mejor acción aprendida (sin exploración)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        scores.append(score)
        print(f"👉 Episodio {episode:2d} | Score: {score}")

    # =========================================================
    # RESULTADOS FINALES
    # =========================================================
    print("\n" + "="*50)
    print(f"RESUMEN FINAL - {ALGO.upper()} (Seed {SEED})")
    print(f"Episodios: {EPISODES}")
    print(f"Recompensa Media: {np.mean(scores):.2f}")
    print(f"Desviación Std:   {np.std(scores):.2f}")
    print(f"Mejor Score:      {np.max(scores)}")
    print("="*50)

    env.close()
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--last", type=int, default=50,
                    help="Número de últimos episodios para calcular métricas")

args = parser.parse_args()

ALGO = args.algo
SEED = args.seed
LAST_N = args.last

path = f"logs/{ALGO}_seed_{SEED}/monitor.csv"

df = pd.read_csv(path, skiprows=1)

rewards = df["r"].values

mean_total = np.mean(rewards)
std_total = np.std(rewards)

mean_last = np.mean(rewards[-LAST_N:])
std_last = np.std(rewards[-LAST_N:])

print("="*50)
print(f"Algoritmo: {ALGO.upper()} | Seed: {SEED}")
print(f"Episodios totales: {len(rewards)}")
print("\n--- MÉTRICAS GLOBALES ---")
print(f"Media total entrenamiento: {mean_total:.2f}")
print(f"Desviación total: {std_total:.2f}")

print(f"\n--- ÚLTIMOS {LAST_N} EPISODIOS ---")
print(f"Media final: {mean_last:.2f}")
print(f"Desviación final: {std_last:.2f}")
print("="*50)
@echo off

echo ==========================================
echo ENTRENANDO DQN SEED 1
echo ==========================================
python train.py --algo dqn --seed 1

echo ==========================================
echo ENTRENANDO DQN SEED 2
echo ==========================================
python train.py --algo dqn --seed 2

echo ==========================================
echo ENTRENANDO DQN SEED 3
echo ==========================================
python train.py --algo dqn --seed 3

echo ==========================================
echo ENTRENANDO DDQN SEED 1
echo ==========================================
python train.py --algo ddqn --seed 1

echo ==========================================
echo ENTRENANDO DDQN SEED 2
echo ==========================================
python train.py --algo ddqn --seed 2

echo ==========================================
echo ENTRENANDO DDQN SEED 3
echo ==========================================
python train.py --algo ddqn --seed 3

echo ==========================================
echo TODOS LOS ENTRENAMIENTOS TERMINARON
echo ==========================================

pause
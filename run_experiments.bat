@echo off
echo Running final experiments...

python train.py --algo dqn --seed 1 --timesteps 1000000 --lr 1e-4 --batch_size 128 --buffer_size 25000 --train_freq 1 --loss huber
python train.py --algo dqn --seed 1 --timesteps 2000000 --lr 1e-4 --batch_size 128 --buffer_size 25000 --train_freq 1 --loss huber
python train.py --algo dqn --seed 1 --timesteps 3000000 --lr 1e-4 --batch_size 128 --buffer_size 25000 --train_freq 1 --loss huber

echo Done
pause
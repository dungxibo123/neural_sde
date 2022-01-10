# bin/sh

python3 run.py --epoch 125 --data svhn --batch-size 3136 --is-ode true --device cuda > "outputs/log_svhn_125e_3136ba_ode.log"
python3 run.py --epoch 125 --data cifar10 --batch-size 3136 --is-ode true --device cuda > "outputs/log_cifar10_125e_3136ba_ode.log"


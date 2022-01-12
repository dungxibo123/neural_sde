# bin/sh

python3 run.py --epoch 200 --data svhn --batch-size 8192 --is-ode true --device cuda > "outputs/log_svhn_200e_8192ba_ode_extend.log"
python3 run.py --epoch 200 --data cifar10 --batch-size 8192 --is-ode true --device cuda > "outputs/log_cifar10_200e_8192ba_ode_extend.log"


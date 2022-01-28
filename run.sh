# bin/sh

python3 run.py --epoch 200 --data svhn --batch-size 8192 --brownian-size 1 --device cuda > "outputs/log_svhn_200e_8192ba_1br_extend.log"
python3 run.py --epoch 200 --data svhn --batch-size 8192 --brownian-size 2 --device cuda > "outputs/log_svhn_200e_8192ba_2br_extend.log"
python3 run.py --epoch 200 --data svhn --batch-size 8192 --brownian-size 3 --device cuda > "outputs/log_svhn_200e_8192ba_3br_extend.log"
python3 run.py --epoch 200 --data cifar10 --batch-size 8192 --brownian-size 1 --device cuda > "outputs/log_cifar10_200e_8192ba_1br_extend.log"
python3 run.py --epoch 200 --data cifar10 --batch-size 8192 --brownian-size 2 --device cuda > "outputs/log_cifar10_200e_8192ba_2br_extend.log"
python3 run.py --epoch 200 --data cifar10 --batch-size 8192 --brownian-size 3 --device cuda > "outputs/log_cifar10_200e_8192ba_3br_extend.log"


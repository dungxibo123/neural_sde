# bin/sh

python3 run.py --epoch 225 --data svhn --batch-size 8192 --brownian-size 1 --device cuda > "outputs/log_svhn_225e_8192ba_1br.log"
python3 run.py --epoch 150 --data svhn --batch-size 4096 --brownian-size 2 --device cuda > "outputs/log_svhn_150e_4096ba_2br.log"
python3 run.py --epoch 150 --data svhn --batch-size 4096 --brownian-size 3 --device cuda > "outputs/log_svhn_150e_4096ba_3br.log"
python3 run.py --epoch 225 --data cifar10 --batch-size 8192 --brownian-size 1 --device cuda > "outputs/log_cifar10_225e_8192ba_1br.log"
python3 run.py --epoch 150 --data cifar10 --batch-size 4096 --brownian-size 2 --device cuda > "outputs/log_cifar10_150e_4096ba_2br.log"
python3 run.py --epoch 150 --data cifar10 --batch-size 4096 --brownian-size 3 --device cuda > "outputs/log_cifar10_150e_4096ba_3br.log"


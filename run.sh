# python train_main.py --exp_name baseline_cifar10_resnet18 --dataset cifar10 --model resnet18 &&
# python train_main.py --exp_name baseline_cifar100_resnet18 --dataset cifar100 --model resnet18 &&

# python train_main.py --exp_name baseline_cifar10_resnet34 --dataset cifar10 --model resnet34 &&
# python train_main.py --exp_name baseline_cifar100_resnet34 --dataset cifar100 --model resnet34



python train_main_sr.py --exp_name baseline_cifar10_resnet18_sr --dataset cifar10 --model resnet18 &&
python train_main_sr.py --exp_name baseline_cifar100_resnet18_sr --dataset cifar100 --model resnet18 &&

python train_main_sr.py --exp_name baseline_cifar10_resnet34_sr --dataset cifar10 --model resnet34 &&
python train_main_sr.py --exp_name baseline_cifar100_resnet34_sr --dataset cifar100 --model resnet34

# CUDA_VISIBLE_DEVICES=7, python train_main.py --exp_name baseline_cifar10_resnet18 --dataset cifar10 --model resnet18
# python train_main.py --exp_name baseline_cifar100_resnet18 --dataset cifar100 --model resnet18 &&

# python train_main.py --exp_name baseline_cifar10_resnet34 --dataset cifar10 --model resnet34 &&
# python train_main.py --exp_name baseline_cifar100_resnet34 --dataset cifar100 --model resnet34


# CUDA_VISIBLE_DEVICES=7, python train_main.py --exp_name baseline_cifar10_resnet18 --dataset cifar10 --model resnet18
# CUDA_VISIBLE_DEVICES=6, python train_main.py --exp_name baseline_cifar100_resnet18_shake --dataset cifar100 --model resnet18_shake

# conda activate copin

######################
CUDA_VISIBLE_DEVICES=7, python train_main.py --exp_name baseline_cifar100_resnet18_randdrop --dataset cifar100 --model resnet18_randdrop --drop True
CUDA_VISIBLE_DEVICES=6, python train_main.py --exp_name baseline_cifar100_resnet34_randdrop --dataset cifar100 --model resnet34_randdrop --drop True

CUDA_VISIBLE_DEVICES=5, python train_main.py --exp_name baseline_cifar100_resnet18_randdrop_dropFalse --dataset cifar100 --model resnet18_randdrop --drop False
CUDA_VISIBLE_DEVICES=4, python train_main.py --exp_name baseline_cifar100_resnet34_randdrop_dropFalse --dataset cifar100 --model resnet34_randdrop --drop False

CUDA_VISIBLE_DEVICES=3, python train_main_shake_distill.py --exp_name baseline_cifar100_resnet18_randdrop_distill --dataset cifar100 --model resnet18_randdrop --drop True
CUDA_VISIBLE_DEVICES=2, python train_main_shake_distill.py --exp_name baseline_cifar100_resnet34_randdrop_distill --dataset cifar100 --model resnet34_randdrop --drop True

CUDA_VISIBLE_DEVICES=1, python train_main_shake_distill_repel.py --exp_name baseline_cifar100_resnet34_randdrop_distill_feature --dataset cifar100 --model resnet34_randdrop



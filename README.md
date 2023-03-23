# MODE

Download decoder.pth and vgg_normalised.pth from https://github.com/naoto0804/pytorch-AdaIN 

main.py is designed for single GPU, and ddp.py is designed for multi-GPU parallelism (currently only for DomainNet). 

python main.py --test_index 3 --dataset PACS --mode A --move_step 10 --gamma 1 --num_mix 4 --model resnet18 --beta 0.4

or

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ddp.py --batch-size 256 --mode A --test_index 2

For rebuttal reference only.

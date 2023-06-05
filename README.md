This repo provides a demo for the ICML 2023 paper "Moderately Distributional Exploration for Domain Generalization" on the PACS dataset.


# MODE

Please download the PACS dataset from https://github.com/MachineLearning2020/Homework3-PACS or http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017 .

Please download decoder.pth and vgg_normalised.pth from https://github.com/naoto0804/pytorch-AdaIN .

Please download resnet18-5c106cde.pth form https://download.pytorch.org/models/resnet18-5c106cde.pth .

Place these downloads in the following directory:

```
MODE/
|--data/
|  |--resnet18-5c106cde.pth
|  |--PACS/
|  |   |--photo/
|  |   |--cartoon/
|  |   |......
|--datalists/......
|--utils/......
|--decoder.pth
|--vgg_normalised.pth
|--ddp.py
|--main.py

```

NOTE: main.py is designed for single GPU, and ddp.py is designed for multi-GPU parallelism (currently only for DomainNet). 

Use commands:
```
python main.py --test_index 3 --dataset PACS --mode A --move_step 10 --gamma 1 --num_mix 4 --model resnet18 --beta 0.4
```

or

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ddp.py --batch-size 256 --mode A --test_index 2
```

Change test_index to determine which domain is the target domain in leave one domain out strategy. 

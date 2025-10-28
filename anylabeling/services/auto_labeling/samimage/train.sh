# train
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12358 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj 05ELD_4 --output work_dirs/hx-25-1_05ELD_4
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12357 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj 2240L_1 --output work_dirs/hx-25-1_2240L_1
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12356 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj B0406B_1 --output work_dirs/hx-25-1_B0406B_1
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12355 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj beiguangjian_3 --output work_dirs/hx-25-1_beiguangjian_3
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12354 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj kuandai_1 --output work_dirs/hx-25-1_kuandai_1


CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12358 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj 05ELD_5 --output work_dirs/hx-25-1_05ELD_5
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12357 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj 2240L_5 --output work_dirs/hx-25-1_2240L_5
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12356 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj B0406B_2 --output work_dirs/hx-25-1_B0406B_2
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12355 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj beiguangjian_7 --output work_dirs/hx-25-1_beiguangjian_7
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12354 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj kuandai_3 --output work_dirs/hx-25-1_kuandai_3


CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12358 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj 05ELD_10 --output work_dirs/hx-25-1_05ELD_10
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12357 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj 2240L_9 --output work_dirs/hx-25-1_2240L_9
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12356 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj B0406B_7 --output work_dirs/hx-25-1_B0406B_7
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12355 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj beiguangjian_10 --output work_dirs/hx-25-1_beiguangjian_10
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12354 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj kuandai_5 --output work_dirs/hx-25-1_kuandai_5




# test
CKPT=xx     # best epoch
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch --master-port 12348 --nproc_per_node=1 train.py --checkpoint pretrained_checkpoint/vit_b.pth --model-type vit_b --obj 05ELD_4 --output work_dirs/hx-25-1_05ELD_4 --eval --restore-model work_dirs/hx-25-1_05ELD_4/epoch_$CKPT.pth

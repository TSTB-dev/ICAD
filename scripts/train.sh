
# run multiple task in parallel
# export CUDA_VISIBLE_DEVICES=2
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/bottle_base_vgg19_ep2000_mblk75_bs64_p4_3.yml --seed 42 & \

# export CUDA_VISIBLE_DEVICES=3
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/bottle_base_vgg19_ep2000_mblk75_bs64_p4_4.yml --seed 42 & \

# export CUDA_VISIBLE_DEVICES=4
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/bottle_base_vgg19_ep2000_mblk75_bs64_p4_5.yml --seed 42 & \

# export CUDA_VISIBLE_DEVICES=0
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/bottle_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=1
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/cable_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=2
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/capsule_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=3
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/carpet_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=4
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/grid_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=5
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/hazelnut_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & 

# export CUDA_VISIBLE_DEVICES=0
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/leather_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=4
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/metal_nut_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \

# # wait for all tasks to finish
# wait

# export CUDA_VISIBLE_DEVICES=0
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/pill_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=4
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/screw_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \

# wait    
# exprot CUDA_VISIBLE_DEVICES=0
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/tile_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=4
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/toothbrush_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \

# wait    
# export CUDA_VISIBLE_DEVICES=0
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/transistor_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=4
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/wood_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=2
# python3 src/training/train_mim.py --config config/training/mvtecad/masking/block/zipper_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \

export CUDA_VISIBLE_DEVICES=0
python3 src/training/train_mim.py --config config/training/mvtecad/backbone/grid_base_resnet50_ep2000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=1
python3 src/training/train_mim.py --config config/training/mvtecad/backbone/screw_base_resnet50_ep2000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=2
python3 src/training/train_mim.py --config config/training/mvtecad/backbone/capsule_base_resnet50_ep2000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=3
python3 src/training/train_mim.py --config config/training/mvtecad/backbone/pill_base_resnet50_ep2000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=4
python3 src/training/train_mim.py --config config/training/mvtecad/backbone/zipper_base_resnet50_ep2000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=5
python3 src/training/train_mim.py --config config/training/mvtecad/backbone/metal_nut_base_resnet50_ep2000_m75.yml --seed 42 & \
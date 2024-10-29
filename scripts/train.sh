
# run multiple task in parallel
export CUDA_VISIBLE_DEVICES=0
python3 src/training/train_mim.py --config config/training/mvtecad/transistor_base_vgg19_ep2000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=1
python3 src/training/train_mim.py --config config/training/mvtecad/wood_base_vgg19_ep2000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=2
python3 src/training/train_mim.py --config config/training/mvtecad/zipper_base_vgg19_ep2000_m75.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=3
# python3 src/training/train_mim.py --config config/training/mvtecad/screw_base_vgg19_ep2000_m75.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=4
# python3 src/training/train_mim.py --config config/training/mvtecad/tile_base_vgg19_ep2000_m75.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=5
# python3 src/training/train_mim.py --config config/training/mvtecad/toothbrush_base_vgg19_ep2000_m75.yml --seed 42 & 
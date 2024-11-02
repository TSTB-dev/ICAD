export CUDA_VISIBLE_DEVICES=0
python src/training/train_mim.py --config config/training/mvtecad/long/capsule_base_vgg19_ep8000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=1
python src/training/train_mim.py --config config/training/mvtecad/long/grid_base_vgg19_ep8000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=2
python src/training/train_mim.py --config config/training/mvtecad/long/pill_base_vgg19_ep8000_m75.yml --seed 42 & \


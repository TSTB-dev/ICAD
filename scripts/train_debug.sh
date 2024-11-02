# (icad) (base) haselab@dl-machine:~/projects/sakai/ICAD$ ls /home/haselab/projects/sakai/ICAD/config/training/mvtecad/stabilize
# capsule_base_vgg19_ep2000_m75.yml  grid_base_vgg19_ep2000_m75.yml  screw_base_vgg19_ep2000_m75.yml

# export CUDA_VISIBLE_DEVICES=0
# python3 src/training/train_mim.py --config config/training/mvtecad/batch/cable_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=1
# python3 src/training/train_mim.py --config config/training/mvtecad/batch/carpet_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=2
# python3 src/training/train_mim.py --config config/training/mvtecad/batch/capsule_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=3
# python3 src/training/train_mim.py --config config/training/mvtecad/batch/grid_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=5
# python3 src/training/train_mim.py --config config/training/mvtecad/batch/pill_base_vgg19_ep2000_mblk75_bs64_p4.yml --seed 42 & \

export CUDA_VISIBLE_DEVICES=0
python3 src/training/train_mim.py --config config/training/mvtecad/stabilize/capsule_base_vgg19_ep2000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=1
python3 src/training/train_mim.py --config config/training/mvtecad/stabilize/grid_base_vgg19_ep2000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=2
python3 src/training/train_mim.py --config config/training/mvtecad/stabilize/screw_base_vgg19_ep2000_m75.yml --seed 42 & \
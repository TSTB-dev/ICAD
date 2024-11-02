
# (icad) (base) haselab@dl-machine:~/projects/ICAD$ ls /home/haselab/projects/ICAD/config/training/mvtecad/batch
# cable_base_vgg19_ep2000_mblk75_bs4_p4.yml    grid_base_vgg19_ep2000_mblk75_bs4_p4.yml
# capsule_base_vgg19_ep2000_mblk75_bs4_p4.yml  pill_base_vgg19_ep2000_mblk75_bs4_p4.yml
# carpet_base_vgg19_ep2000_mblk75_bs4_p4.yml

# run multiple task in parallel
export CUDA_VISIBLE_DEVICES=0
python3 src/training/train_eliminator.py --config config/training/mvtecad/elim/elim_capsule_base_vgg19_ep4000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=1
python3 src/training/train_eliminator.py --config config/training/mvtecad/elim/elim_pill_base_vgg19_ep4000_m75.yml --seed 42 & \
export CUDA_VISIBLE_DEVICES=2
python3 src/training/train_eliminator.py --config config/training/mvtecad/elim/elim_grid_base_vgg19_ep4000_m75.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=3
# python3 src/training/train_mim.py --config config/training/mvtecad/long/grid_base_vgg19_ep4000_m75.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=3
# python3 src/training/train_mim.py --config config/training/mvtecad/batch/grid_base_vgg19_ep2000_mblk75_bs4_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=3
# python3 src/training/train_mim.py --config config/training/mvtec_loco/pdn_small/splicing_connectors_base_pdns_ep2000_m75_bs64_p4.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=4
# python3 src/training/train_mim.py --config config/training/mvtec_loco/tile_base_vgg19_ep2000_m75.yml --seed 42 & \
# export CUDA_VISIBLE_DEVICES=5
# python3 src/training/train_mim.py --config config/training/mvtec_loco/toothbrush_base_vgg19_ep2000_m75.yml --seed 42 & 
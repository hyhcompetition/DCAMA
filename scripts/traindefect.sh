python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=16005 \
./train.py --datapath "../data" \
           --benchmark defect \
           --fold 0 \
           --bsz 4 \
           --nworker 8 \
           --backbone swin \
           --feature_extractor_path "../model/swin_base_patch4_window12_384.pth" \
           --logpath "./logs" \
           --lr 1e-3 \
           --nepoch 150
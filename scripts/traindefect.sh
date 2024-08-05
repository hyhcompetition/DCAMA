torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=16005 \
./train.py --datapath "/root/autodl-tmp/data_patch" \
           --benchmark defect \
           --fold 0 \
           --bsz 12 \
           --nworker 8 \
           --backbone swin \
           --feature_extractor_path "/root/autodl-tmp/Model/swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "/root/autodl-tmp/logs/DCAMA_patch"
           --lr 1e-3 \
           --nepoch 250 \
	   --load ''

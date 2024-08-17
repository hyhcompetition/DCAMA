torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=16005 \
./train.py --datapath "../data_patch" \
           --benchmark defect \
           --fold 0 \
           --bsz 6 \
           --nworker 8 \
           --backbone swin \
           --feature_extractor_path "swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "../logs/DCAMA_PATCH_FINETINE"
           --lr 1e-3 \
           --nepoch 250 \
	   --load 'fold_0_0802_185050/best_model.pt'

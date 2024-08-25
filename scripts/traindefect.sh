torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=16005 \
./train.py --datapath "../../Dataset/VISION24-data-challenge-final-test/final_test_patch_cls" \
           --benchmark defect \
           --fold 0 \
           --bsz 4 \
           --nworker 8 \
           --backbone swin \
           --feature_extractor_path "../../ModelPth/DCAMA/swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "../logs/DCAMA_PATCH_FINETINE"
           --lr 1e-3 \
           --nepoch 250 \
	       --load '' \
           --global_path ""\
           --local_path ""

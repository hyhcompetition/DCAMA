python ./test.py --datapath "/root/autodl-tmp/data_new" \
                 --benchmark defect \
                 --fold 0 \
                 --bsz 1 \
                 --nworker 8 \
                 --backbone swin \
                 --feature_extractor_path "" \
                 --logpath "./logs" \
                 --load "./best_model.pt" \
                 --nshot 1 \
                 --vispath "/root/autodl-tmp/logs/DCAMA/vis" \
                 --visualize


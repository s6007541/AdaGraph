# export LD_LIBRARY_PATH=/home/sorn111930/anaconda3/envs/gtrans2/lib/


# python train_both_all.py --gpu_id=0 --dataset=cora --model=GCN  --seed=0 --tune=0\
#     --finetune=1 --tent=1 --use_learned_stat=1 --bn_momentum=0.3 --lr_feat 0.01

# cora best
# python train_both_all.py --gpu_id=0 --dataset=cora --model=GCN  --seed=0 --tune=0\
#     --finetune=1 --tent=1 --use_learned_stat=0 --bn_momentum 0.0\
#    --ep_tta 1 --lr_tta 0.01\
# # --sam  --ent_filter 0.5 

#amz-photo best
# python train_both_all.py --gpu_id=0 --dataset=amazon-photo --model=GCN  --seed=0 --tune=0\
#     --finetune=1 --tent=1 --use_learned_stat=0 --bn_momentum 0.0\
#    --ep_tta 1 --lr_tta 0.01\
#     # --sam  --ent_filter 0.5 

#ti
python train_both_all.py --gpu_id=0 --dataset=twitch-e --model=GCN  --seed=0 --tune=0\
    --finetune=1 --tent=1 --use_learned_stat=0 --bn_momentum 0.0\
  --ep_tta 100 --lr_tta 0.0001 --ent_filter 0.1
#   --sam  --ent_filter 0.5 
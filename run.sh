export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,2
# export ACCELERATE_USE_DDP_FIND_UNUSED_PARAMETERS=true
torchrun \
  --nproc_per_node 2 \
  --master_port 29550 \
  Colon_train_seg.py
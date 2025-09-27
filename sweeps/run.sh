NGPU=${NGPU:-"1"}
PYTORCH_ALLOC_CONF="expandable_segments:True" \

uv run torchrun --local-ranks-filter 0 --nproc_per_node=${NGPU} src/zeroband/train.py "$@"
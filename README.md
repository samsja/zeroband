# zeroband

## install

```bash
curl -sSL https://raw.githubusercontent.com/samsja/zeroband/main/install.sh | bash
```

## run

```bash
uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/zeroband/train.py @ configs/debug.toml
```

```bash
uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/zeroband/train.py @ configs/debug.toml --data.name allenai/c4
```

```bash
uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/zeroband/train.py @ configs/14M.toml
```

## sweeps

```bash
wandb sweep sweeps/14M/bs-256-steps-2k.yaml
```


##

codebase to create low bandwith scaling law.


TODO:

- [x] add llama
- [ ] add mup to llama
- [x] add full graph compile ddp
- [ ] add low precision all reduce support
- [ ] add dion low rank support
- [ ] add diloco with ddp support

TODO slow

- [x] add data
- [x] add lr scheduler 
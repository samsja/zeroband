# zeroband

```bash
uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/zeroband/train.py @ configs/debug.toml
```


##

codebase to create low bandwith scaling law.


TODO:

- [x] add llama
- [ ] add mup to llama
- [ ] add full graph compile ddp
- [ ] add low precision all reduce support
- [ ] add dion low rank support
- [ ] add diloco with ddp support


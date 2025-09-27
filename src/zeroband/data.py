import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer

from zeroband.config import DataConfig
from zeroband.utils import World


class HfDataset(IterableDataset):
    def __init__(self, dataset_name: str, tokenizer: PreTrainedTokenizer, seq_len: int):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.target_seq_len = seq_len + 1

        world = World()

        ds = load_dataset(dataset_name, split="train", streaming=True)
        self._data = split_dataset_by_node(ds, world.world_size, world.rank)

        self._current_batch = []

    def __iter__(self):
        while True:
            sample = self._data.next()["text"]
            sample = self.tokenizer.encode(sample, add_special_tokens=False, add_bos=True, add_eos=True)[
                : self.target_seq_len
            ]

            self._current_batch.append(sample)

            if len(self._current_batch) >= self.target_seq_len:  # always need +1 on seq len for labels
                batch_to_yield = torch.tensor(self._current_batch[: self.target_seq_len])
                self._current_batch = []
                yield {"input_ids": batch_to_yield[:-1], "labels": batch_to_yield[1:]}

    def load_state_dict(self, state_dict):
        assert state_dict.keys() == ["token_buffer", "data"]
        self._current_batch = state_dict["token_buffer"]
        self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        return {"token_buffer": self._token_buffer, "data": self._data.state_dict()}


class FakeDataset(IterableDataset):
    def __init__(self, config: DataConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def __iter__(self):
        while True:
            input_ids = torch.randint(0, len(self.tokenizer), (self.config.seq_len + 1,)).long()
            yield {"input_ids": input_ids[:-1], "labels": input_ids[1:]}

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict): ...


def setup_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer):
    if config.fake:
        dataset = FakeDataset(config, tokenizer)
    else:
        dataset = HfDataset(config.name, tokenizer, config.seq_len)
    return StatefulDataLoader(dataset, batch_size=config.micro_batch_size, num_workers=config.num_workers)

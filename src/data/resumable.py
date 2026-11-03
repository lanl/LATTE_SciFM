"""Deterministic distributed batch order with a consumer-owned resume cursor."""
from itertools import islice

from torch.utils.data import DistributedSampler, Sampler


class ResumableDistributedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, world_size=1, rank=0, seed=0):
        self.batch_size = int(batch_size)
        if self.batch_size < 1: raise ValueError("batch_size must be positive")
        self.sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        self.batches_per_epoch = len(self.sampler) // self.batch_size
        if not self.batches_per_epoch:
            raise ValueError("Dataset is too small for one complete distributed batch")
        self.specification = {
            "version": 1, "dataset_length": len(dataset), "batch_size": self.batch_size,
            "world_size": int(world_size), "seed": int(seed),
            "dataset_signature": repr((type(dataset).__name__, getattr(dataset, "data_stream_id", None),
                                       getattr(dataset, "delta_t_choices", None))),
        }
        self.epoch = 0
        self.batch_offset = 0

    def __iter__(self):
        epoch, offset = self.epoch, self.batch_offset
        self.sampler.set_epoch(epoch)
        indices = islice(iter(self.sampler), offset * self.batch_size, None)
        for batch_index in range(offset, self.batches_per_epoch):
            yield list(islice(indices, self.batch_size))

    def __len__(self):
        return self.batches_per_epoch - self.batch_offset

    def advance(self):
        self.batch_offset += 1
        if self.batch_offset == self.batches_per_epoch:
            self.epoch += 1
            self.batch_offset = 0

    def seek(self, consumed_batches):
        self.epoch, self.batch_offset = divmod(int(consumed_batches), self.batches_per_epoch)

    def state_dict(self):
        return dict(self.specification, epoch=self.epoch, batch_offset=self.batch_offset)

    def load_state_dict(self, state):
        for key, expected in self.specification.items():
            if state.get(key) != expected:
                raise ValueError(f"Data stream mismatch for {key}: saved={state.get(key)!r}, current={expected!r}")
        epoch, offset = int(state["epoch"]), int(state["batch_offset"])
        if epoch < 0 or not 0 <= offset < self.batches_per_epoch:
            raise ValueError("Invalid saved data-stream position")
        self.epoch, self.batch_offset = epoch, offset

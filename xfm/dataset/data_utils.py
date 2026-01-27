from typing import Any
from streaming import StreamingDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
import numpy as np
import torch
import random


class StreamingWrapperDataset(StreamingDataset):
    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        vae_flat = sample["vae_output"]
        vae_tensor = torch.from_numpy(vae_flat).reshape(4, 32, 32) * 0.13025
        label = int(sample["label"])
        return vae_tensor, label

class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0

_encodings["uint8"] = uint8


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def imagenet_worker_init(worker_id):
    seed_worker(worker_id)
    from streaming.base.format.mds.encodings import _encodings
    _encodings["uint8"] = uint8
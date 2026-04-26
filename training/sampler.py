import random
from torch.utils.data import Sampler


class BucketShuffleSampler(Sampler):
    def __init__(
        self,
        lengths: list[float],
        num_buckets: int = 100,
        batch_size: int = 16,
        shuffle: bool = True,
    ):
        self.lengths = lengths
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            random.shuffle(indices)

        sorted_indices = sorted(indices, key=lambda i: self.lengths[i])

        bucket_size = max(1, len(sorted_indices) // self.num_buckets)
        buckets = []
        for i in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[i : i + bucket_size]
            if self.shuffle:
                random.shuffle(bucket)
            buckets.extend(bucket)

        return iter(buckets)

    def __len__(self):
        return len(self.lengths)

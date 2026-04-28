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


class DynamicBatchSampler(Sampler):
    def __init__(
        self,
        lengths: list[float],
        max_tokens: int,
        frames_per_sec: float = 41.0,
        max_batch_size: int = 512,
        min_batch_size: int = 4,
        num_buckets: int = 100,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.frames_per_sec = frames_per_sec
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._batches = self._make_batches()

    def _make_batches(self):
        indices = list(range(len(self.lengths)))

        sorted_indices = sorted(indices, key=lambda i: self.lengths[i])

        bucket_size = max(1, len(sorted_indices) // self.num_buckets)
        buckets = []
        for i in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[i : i + bucket_size]
            buckets.append(bucket)

        batches = []
        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)

            current_batch = []
            current_tokens = 0

            for idx in bucket:
                n_frames = max(1, int(self.lengths[idx] * self.frames_per_sec))
                n_tokens = n_frames

                if current_batch and (
                    (current_tokens + n_tokens) > self.max_tokens
                    or len(current_batch) >= self.max_batch_size
                ):
                    if len(current_batch) >= self.min_batch_size or not self.drop_last:
                        batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0

                current_batch.append(idx)
                current_tokens += n_tokens

            if current_batch:
                if len(current_batch) >= self.min_batch_size or not self.drop_last:
                    batches.append(current_batch)

        if self.shuffle:
            random.shuffle(batches)

        return batches

    def __iter__(self):
        if self.shuffle:
            self._batches = self._make_batches()
        for batch in self._batches:
            yield batch

    def __len__(self):
        return len(self._batches)

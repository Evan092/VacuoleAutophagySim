import random
from torch.utils.data import Sampler

class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.maxSteps = 50
        self.threshold   = len(dataset) - self.maxSteps  # i.e. 80 if total is 100
        self.shuffle = shuffle

    def __iter__(self):
        # Shuffle all indices each epoch
        all_idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(all_idxs)

        for i in range(0, len(all_idxs), self.batch_size):
            batch_idxs = all_idxs[i:i + self.batch_size]
            n = random.randint(1, self.maxSteps)  # fixed horizon per batch

            batch_keys = []
            for idx in batch_idxs:
                idx0 = idx - self.maxSteps if idx >= self.threshold else idx
                batch_keys.append((idx0, idx0 + n))
            yield batch_keys

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

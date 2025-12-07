import random
from torch.utils.data import Sampler

class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle, drop_last=False):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.frequency = 50
        self.maxSteps = 2
        self.threshold   = len(dataset)  # i.e. 80 if total is 100
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Shuffle all indices each epoch
        all_idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(all_idxs)

        for i in range(0, len(all_idxs), self.batch_size):
            batch_idxs = all_idxs[i:i + self.batch_size]
            if self.drop_last and len(batch_idxs) < self.batch_size:
                continue

            nSteps = random.randint(1, self.maxSteps) * self.frequency+200  # fixed horizon per batch

            batch_keys = []
            for idx in batch_idxs:
                outputNumber = random.randint(1, 5)
                startStep = 0
                batch_keys.append((idx, outputNumber, startStep, nSteps))
            yield batch_keys


    #def __len__(self):
        #return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __len__(self):
        n = (len(self.dataset) + self.batch_size - 1)
        full, rem = divmod(n, self.batch_size)
        return full if self.drop_last else (full + (1 if rem else 0))
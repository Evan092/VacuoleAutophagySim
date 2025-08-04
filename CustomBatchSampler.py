import random
from torch.utils.data import Sampler

class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.frequency = 50
        self.maxSteps = 6
        self.threshold   = len(dataset)  # i.e. 80 if total is 100
        self.shuffle = shuffle

    def __iter__(self):
        # Shuffle all indices each epoch
        all_idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(all_idxs)

        for i in range(0, len(all_idxs), self.batch_size):
            batch_idxs = all_idxs[i:i + self.batch_size]
            nSteps = random.randint(1, self.maxSteps)*self.frequency  # fixed horizon per batch

            batch_keys = []
            for idx in batch_idxs:

                outputNumber = random.randint(1, 5)

                startStep = 0


                batch_keys.append((idx, outputNumber, startStep, nSteps))
            yield batch_keys

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

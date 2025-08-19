import math
import torch
from torch.utils.data import Sampler, Dataset

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset : Dataset, batch_size : int, k : int, num_replicas : int, rank : int, seed :int = 0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch : int):
        self.epoch = epoch  # Used to synchronize random state across epochs




class DistributedKRepeatSamplerNew(Sampler):
    def __init__(self, dataset : Dataset, batch_size : int, k : int, num_replicas : int, rank : int, seed :int = 0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas, process num, gpu num
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        if self.total_samples % self.k == 0:
           # k | n*b, which means all repetitions of one sample can be yielded in one iteration
           assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
           self.m = self.total_samples // self.k  # Number of unique samples
           self.min_iter = 1
        else:
            # k does not divide n*b, which means we need to yield some samples in multiple iterations
            # Find the least common multiple (LCM) of k and n*b
            lcm = math.lcm(self.k, self.total_samples)
            self.m = lcm // self.k  # Number of unique samples
            self.min_iter = lcm // self.total_samples
            self.total_samples = lcm  # Adjust total samples to LCM for uniformity

        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate m*k total samples.
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            for it in range(self.min_iter):
                offset = it * self.batch_size * self.num_replicas
                per_card_samples = []
                for i in range(self.num_replicas):
                    start = i * self.batch_size + offset
                    end = start + self.batch_size + offset
                    per_card_samples.append(shuffled_samples[start:end])
                
                # Return current replica's sample indices
                yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch : int):
        self.epoch = epoch  # Used to synchronize random state across epochs




if __name__ == "__main__":
    from flow_grpo.datasets.prompt_dataset import TextPromptDataset
    train_batch_size = 1
    num_image_per_prompt = 4
    num_processes = 4

    dataset = 'dataset/ocr'
    train_dataset = TextPromptDataset(dataset, 'train')
    train_samplers = [
        DistributedKRepeatSamplerNew( 
        dataset=train_dataset,
        batch_size=train_batch_size,
        k=num_image_per_prompt,
        num_replicas=num_processes,
        rank=i,
        seed=42)
        for i in range(num_processes)
    ]

    for epoch in range(3):
        for sampler in train_samplers:
            sampler.set_epoch(epoch)
        
        for i, sample_indices in enumerate(train_samplers[0]):
            print(f"Epoch {epoch}, Process {i}: Sample Indices: {sample_indices}")
            if i == 5:  # Limit to 5 iterations for demonstration
                break

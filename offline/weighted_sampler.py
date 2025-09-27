import torch
from torch.utils.data import WeightedRandomSampler

# Create phase-based weights
weights = torch.where(phases == 2, 2.0, torch.where(phases == 1, 1.0, 0.5))

# Weighted sampler for training
sampler = WeightedRandomSampler(weights, num_samples=50_000, replacement=True)

# Test sampling
sample_indices = list(iter(sampler))[:10]
print("Sampled indices:", sample_indices)
print("Phases sampled:", phases[sample_indices])


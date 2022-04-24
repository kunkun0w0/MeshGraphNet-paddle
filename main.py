from dataset import SimpleFlag
from paddle.io import DataLoader

test_dataset = SimpleFlag.iterator_dataset(dataset_dir='./data/flag_simple',
                                          split='test',
                                          noise_scale=0.003,
                                          noise_gamma=0.1)


dataloader = DataLoader(
    test_dataset,
    num_workers=1,
    batch_size=4,
    drop_last=True)

for i in dataloader:
    print(i.shape)


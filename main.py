import paddle

from dataset import SimpleFlag
from paddle.io import DataLoader
from dataset.DataProcess import h5py_to_tensor

paddle.set_device('gpu')

test_dataset = SimpleFlag.simple_flag(dataset_dir='./data/flag_simple',
                                          split='test',
                                          noise_scale=0.003,
                                          noise_gamma=0.1)


dataloader = DataLoader(
    test_dataset,
    num_workers=4,
    batch_size=1,
    drop_last=True)

for idx, data in enumerate(dataloader):
    tmp = h5py_to_tensor(data)
    print(idx)


import paddle
from dataset.simple_flag_tf import FlagSimpleDataset
# from dataset import SimpleFlag
# from paddle.io import DataLoader
from dataset.mesh_loader import data_to_feature
#
#
# test_dataset = SimpleFlag.simple_flag(dataset_dir='./data/flag_simple',
#                                       split='test',
#                                       noise_scale=0.003,
#                                       noise_gamma=0.1)
#
# dataloader = DataLoader(test_dataset, num_workers=4, batch_size=2, drop_last=True)
#
# for idx, item in enumerate(dataloader):
#     node_features, edge_features, senders, receivers, frame = h5py_to_tensor(item)
#
#
#
#     # print(node_features.shape)
#     # print(edge_features.shape)
#     # print(senders.shape)
#     # print(receivers.shape)
#     # print(len(frame))
#     #
#     # print()
#     print(idx)


tf_record = FlagSimpleDataset(path='./data/flag_simple/', split='test')
for id, data in enumerate(tf_record):
    data_to_feature(data)
    print(id)
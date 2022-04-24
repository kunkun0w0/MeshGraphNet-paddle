import paddle
from paddle.io import Dataset

import os
import os.path as osp
import numpy as np
import h5py

import common


def add_targets(trajectory, fields, add_history):
    out = {}
    for key in trajectory.keys():
        val = trajectory[key]
        out[key] = val[1:-1]
        if key in fields:
            out[f'target|{key}'] = val[2:]
            if add_history:
                out[f'prev|{key}'] = val[0:-2]
    return out


def add_noise(frame, fields, scale, gamma):
    if scale == 0:
        return frame

    for field in fields:
        noise = paddle.tensor.random.gaussian(frame[field].shape, std=scale, dtype=paddle.float32)

        # don't apply noise to boundary nodes
        mask = paddle.equal(frame['node_type'], common.NodeType.NORMAL)
        noise = paddle.where(mask, noise, paddle.zeros_like(noise))

        frame[field] += noise
        frame[f'target|{field}'] += (1.0 - gamma) * noise

    return frame


class iterator_dataset(Dataset):
    def __init__(self, dataset_dir, noise_scale, noise_gamma,
                 split='train', fields=None, add_history=True):
        super(iterator_dataset, self).__init__()
        #  Simple Flag attributes
        self.data_keys = ("cells", "mesh_pos", "node_type", "world_pos")
        self.out_keys = list(self.data_keys) + ['time']
        self.train_len = 1000
        self.test_len = 100
        self.valid_len = 100
        self.time_interval = 0.02

        # dataset configs
        self.split = split
        assert split in ('train', 'test', 'valid'), "Wrong split input!"
        dataset_dir = osp.join(dataset_dir, split + '.h5')
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.add_history = add_history
        if fields is None:
            self.fields = ['world_pos']
        else:
            self.fields = fields

        for field in self.fields:
            assert field in self.data_keys, "Wrong fields input!"

        # load data
        self.noise_scale = noise_scale
        self.noise_gamma = noise_gamma

        print('Dataset ' + dataset_dir.split('/')[-2] + ' Initialized')

    def open_hdf5(self):
        self.file_handle = h5py.File(self.dataset_dir, 'r')

    def __getitem__(self, idx):
        if not hasattr(self, 'file_handle'):
            self.open_hdf5()

        data = self.file_handle[str(idx)]
        data = add_targets(trajectory=data, fields=self.fields, add_history=self.add_history)
        if self.split == 'train':
            data = add_noise(frame=data, fields=self.fields, scale=self.noise_scale, gamma=self.noise_gamma)

        if 'velocity' in data.keys():
            velocity = data['velocity']
        else:
            velocity = data['world_pos'] - data['prev|world_pos']

        velocity = paddle.to_tensor(np.array(velocity))
        node_type = paddle.to_tensor(np.squeeze(np.array(data['node_type'][:, 0])))
        node_type = paddle.nn.functional.one_hot(node_type, num_classes=common.NodeType.SIZE)
        node_type = paddle.expand(paddle.unsqueeze(node_type, axis=1),
                                  shape=[node_type.shape[0], velocity.shape[1], node_type.shape[1]])
        node_features = paddle.concat([velocity, node_type], axis=-1)

        # construct graph edges
        senders, receivers = common.triangles_to_edges(paddle.to_tensor(np.array(data['cells'])))
        relative_world_pos = (paddle.gather(data['world_pos'], senders) -
                              paddle.gather(data['world_pos'], receivers))
        relative_mesh_pos = (paddle.gather(data['mesh_pos'], senders) -
                             paddle.gather(data['mesh_pos'], receivers))
        edge_features = paddle.concat([
            relative_world_pos,
            paddle.norm(relative_world_pos, axis=-1, keepdim=True, p=2),
            relative_mesh_pos,
            paddle.norm(relative_mesh_pos, axis=-1, keepdim=True, p=2)], axis=-1)

        return node_features, edge_features, senders, receivers, data

    def __len__(self):
        if self.split == 'train':
            return self.train_len
        elif self.split == 'test':
            return self.test_len
        elif self.split == 'valid':
            return self.valid_len
        else:
            print("with wrong split!")
            raise RuntimeError

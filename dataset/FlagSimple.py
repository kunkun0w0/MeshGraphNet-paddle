import paddle

from paddle.io import Dataset
from tfrecord.paddle.dataset import TFRecordDataset
import os
import json
import numpy as np

from common import NodeType


class dataset(Dataset):
    def __init__(self, path, split, add_history=True,
                 noise_scale=0.003, noise_gamma=0.1, fields=None):
        super(dataset, self).__init__()
        if fields is None:
            fields = ['world_pos']

        self.path = path
        self.split = split

        try:
            with open(os.path.join(path, 'meta.json'), 'r') as fp:
                self.meta = json.loads(fp.read())
            self.shapes = {}
            self.dtypes = {}
            self.types = {}
            for key, field in self.meta['features'].items():
                self.shapes[key] = field['shape']
                self.dtypes[key] = field['dtype']
                self.types[key] = field['type']
        except FileNotFoundError as e:
            print(e)
            quit()

        tfrecord_path = os.path.join(path, split + ".tfrecord")
        index_path = os.path.join(path, split + ".id")
        tf_dataset = TFRecordDataset(tfrecord_path, index_path, None)
        loader = paddle.io.DataLoader(tf_dataset, batch_size=1)

        self.noise_scale = noise_scale
        self.noise_gamma = noise_gamma
        self.fields = fields
        self.add_history = add_history

        self.dataset = list(iter(loader))

    def __len__(self):
        # flag simple dataset contains 1000 trajectories, each trajectory contains 400 steps
        if self.split == 'train':
            return 1000
        elif self.split == 'valid':
            return 100
        elif self.split == 'test':
            return 100

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        trajectory = {}
        # decode bytes into corresponding dtypes
        for key, value in sample.items():
            raw_data = value.numpy().tobytes()
            mature_data = np.frombuffer(raw_data, dtype=getattr(np, self.dtypes[key]))
            mature_data = paddle.to_tensor(mature_data)
            reshaped_data = paddle.reshape(mature_data, self.shapes[key])
            if self.types[key] == 'static':
                reshaped_data = paddle.tile(reshaped_data, (self.meta['trajectory_length'], 1, 1))
            elif self.types[key] == 'dynamic_varlen':
                pass
            elif self.types[key] != 'dynamic':
                raise ValueError('invalid data format')
            trajectory[key] = reshaped_data

        trajectory = self.add_targets(trajectory)
        trajectory = self.add_noise(trajectory)

        if 'velocity' not in trajectory.keys():
            trajectory['velocity'] = trajectory['world_pos'] - trajectory['prev|world_pos']

        return trajectory

    def add_targets(self, trajectory):
        """Adds target and optionally history fields to dataframe."""
        fields = self.fields
        add_history = self.add_history

        out = {}

        for key, val in trajectory.items():
            out[key] = val[1:-1]
            if key in fields:
                if add_history:
                    out['prev|' + key] = val[0:-2]
                out['target|' + key] = val[2:]

        return out

    def add_noise(self, trajectory):
        fields = self.fields
        scale = self.noise_scale
        gamma = self.noise_gamma

        if self.split == 'train' and scale != 0:
            for field in fields:
                noise = paddle.tensor.random.gaussian(trajectory[field].shape, std=scale)

                # don't apply noise to boundary nodes
                mask = paddle.equal(trajectory['node_type'],
                                    paddle.to_tensor(int(NodeType.NORMAL), dtype=paddle.int32))
                noise = paddle.where(mask, noise, paddle.zeros_like(noise))

                trajectory[field] += noise
                trajectory[f'target|{field}'] += (1.0 - gamma) * noise

        return trajectory

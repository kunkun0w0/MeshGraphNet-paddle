# https://github.com/echowve/meshGraphNets_pytorch/blob/master/parse_tfrecord.py

# tfrecord -> hdf5
import tensorflow as tf
import functools
import json
import os
import numpy as np
import h5py
import sys
from tqdm import tqdm


def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                     for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + '.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


if __name__ == '__main__':
    tf.enable_resource_variables()
    tf.enable_eager_execution()

    if len(sys.argv) < 3:
        print("Usage: tfrecord2idx <tfrecord path> <index path>")
        sys.exit()

    tfrecord_path = sys.argv[1]
    h5py_path = sys.argv[2]

    os.makedirs(h5py_path, exist_ok=True)

    for split in ['train', 'test', 'valid']:
        print()
        print(f"###### Start {split}! ######")

        ds = load_dataset(tfrecord_path, split)
        f_name = split + '.h5'
        save_path = os.path.join(h5py_path, f_name)
        f = h5py.File(save_path, "w")
        print(save_path)

        for index, d in enumerate(ds):
            g = f.create_group(str(index))

            for k in list(d.keys()):
                g[k] = d[k].numpy()

            print(index, end='\r')
        f.close()

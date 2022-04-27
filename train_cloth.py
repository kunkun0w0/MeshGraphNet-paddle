import os
import pickle
import argparse

from tqdm import tqdm
import numpy as np
import paddle
from paddle.optimizer import Adam
from visualdl import LogWriter

import common
from model.model import EdgeSet
from model.model import MultiGraph

from model import simulator_cloth
from dataset import mesh_loader
# from evaluate import rollout

import datetime


# def validation(model, dataset, num_trajectories=5):
#     print('\nEvaluating...')
#     horizons = [1, 10, 20, 50, 100, 200, 398]
#     all_errors = {horizon: [] for horizon in horizons}
#     for i, trajectory in enumerate(dataset.take(num_trajectories)):
#         initial_frame = {k: v[0] for k, v in trajectory.items()}
#         predicted_trajectory = rollout(model, initial_frame, trajectory['cells'].shape[0])
#
#         error = paddle.mean(paddle.square(predicted_trajectory - trajectory['world_pos']), axis=-1)
#         for horizon in horizons:
#             all_errors[horizon].append(paddle.sqrt(paddle.mean(error[1:horizon + 1])).numpy())
#
#     return {k: np.mean(v) for k, v in all_errors.items()}


def train(data_path=os.path.join(os.path.dirname(__file__), 'data', 'flag_simple'),
          epoch=10000, save_epoch=100, val_epoch=10, save_path='./model_save/flag_simple',
          checkpoint=None):
    train_dataset = mesh_loader.load_dataset(
        path=data_path,
        split='train',
        fields=['world_pos'],
        add_history=True,
        noise_scale=0.003,
        noise_gamma=0.1
    )

    # valid_dataset = mesh_loader.load_dataset(
    #     path=data_path,
    #     split='valid',
    #     fields=['world_pos'],
    #     add_history=True
    # )

    train_len = len(train_dataset)
    # valid_len = len(valid_dataset)

    model = simulator_cloth.SimulatorCloth(
        node_input_size=12,
        edge_input_size=7,
        output_size=2,
        num_iterations=15,
        num_edge_types=1
    )

    # todo: add paddle.optimizer.lr.ExponentialDecay
    # scheduler  = paddle.optimizer.lr.ExponentialDecay(learning_rate=1e-4, gamma=paddle.pow(0.1, -num_steps // 2))
    # optimizer = Adam(learning_rate=scheduler, parameters=model.parameters())
    optimizer = Adam(learning_rate=1e-4, parameters=model.parameters())

    if checkpoint:
        state_dict = paddle.load(checkpoint)
        opt_state = state_dict['opt_state']
        model_state = state_dict['model_state']
        optimizer.set_state_dict(opt_state)
        model.set_state_dict(model_state)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = './Visualization/logs/' + current_time
    os.makedirs(log_dir, exist_ok=True)

    train_log_dir = log_dir + '/train'
    val_log_dir = log_dir + '/validation'

    train_summary_writer = LogWriter(train_log_dir)
    val_summary_writer = LogWriter(val_log_dir)

    for e in range(epoch):
        print(f"Epoch {e + 1}:")
        for idx, data in enumerate(train_dataset):
            node_features, edge_features, senders, receivers, frame = mesh_loader.h5py_to_tensor(data)
            graph = MultiGraph(node_features, edge_sets=[EdgeSet(edge_features, senders, receivers)])

            output, target_normalized, acceleration = model(graph=graph, frame=frame)
            loss = simulator_cloth.cloth_loss(output, target_normalized, acceleration)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            train_summary_writer.add_scalar("train loss", loss.numpy().item(), e * train_len + idx)

            if idx % 10 == 0:
                print(f"step:{idx} => loss:{loss.numpy().item()}")

        if (e + 1) % save_epoch == 0:
            opt_state = optimizer.state_dict()
            model_state = model.state_dict()
            state_dict = {"opt_state": opt_state, "model_state": model_state}
            paddle.save(state_dict, path=os.path.join(save_path, str(e + 1) + '.pdparams'))

        # # perform validation
        # if (e + 1) % val_epoch == 0:
        #     errors = validation(model, valid_dataset)
        #     for k, v in errors.items():
        #         val_summary_writer.add_scalar(f'validation {k}-rmse', v, step=e + 1)
        #     print(', '.join([f'{k}-step RMSE: {v}' for k, v in errors.items()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", default=None, help="Path to checkpoint file used to resume training")
    parser.add_argument("--data_path", default=os.path.join(os.path.dirname(__file__), 'data', 'flag_simple'),
                        help="Path to dataset")
    parser.add_argument("--epoch", type=int, default=1000, help="Number of epochs to train (default :1000)")
    parser.add_argument("--save_epoch", type=int, default=100, help="Every n epochs to save (default :100)")
    parser.add_argument("--save_path", default='./model_save/flag_simple', help="Path to save")
    parser.add_argument("--val_epoch", type=int, default=10, help="Every n epochs to valid (default :10)")

    args = parser.parse_args()
    train(data_path=args.data_path, epoch=args.epoch, save_epoch=args.save_epoch, val_epoch=args.val_epoch,
          save_path=args.save_path, checkpoint=args.checkpoint)


if __name__ == '__main__':
    paddle.device.set_device("gpu:1")
    main()

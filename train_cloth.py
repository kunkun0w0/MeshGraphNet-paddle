import os
import pickle
import argparse

import numpy as np
from tqdm import tqdm
import paddle
from paddle.optimizer import Adam
from visualdl import LogWriter

from model.model import EdgeSet
from model.model import MultiGraph

from model import simulator_cloth
from model.simulator_cloth import cloth_loss, cloth_predict
from dataset import mesh_loader

import datetime


def train(data_path=os.path.join(os.path.dirname(__file__), 'data', 'flag_simple'),
          steps=1000000, val_epoch=10, save_path='./model_save/flag_simple',
          checkpoint=None, frames=1):
    train_dataset = mesh_loader.load_dataset(
        path=data_path,
        split='train',
        fields=['world_pos'],
        add_history=True,
        noise_scale=0.003,
        noise_gamma=0.1,
    )

    valid_dataset = mesh_loader.load_dataset(
        path=data_path,
        split='valid',
        fields=['world_pos'],
        add_history=True,
    )

    train_len = len(train_dataset)
    # valid_len = len(valid_dataset)

    model = simulator_cloth.SimulatorCloth(
        node_input_size=12,
        edge_input_size=7,
        output_size=3,
        num_iterations=15,
        num_edge_types=1
    )

    model.train()

    # todo: add paddle.optimizer.lr.ExponentialDecay
    optimizer = Adam(learning_rate=3e-4, parameters=model.parameters())

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

    # train
    flag = 0
    for e in range(10):
        if flag:
            break

        print(f"Epoch {e + 1} train")
        model.train()
        loop = tqdm(train_dataset, ncols=100)
        for idx, data in enumerate(loop):
            node_features, edge_features, senders, receivers, frame = mesh_loader.data_to_feature(data)
            # split frame
            t, nodes, c = node_features.shape
            loss_value = []
            for i in range(0, t, frames):
                current_step = (e * train_len + idx) * t + i
                if current_step > steps:
                    flag = 1
                    break

                if i + frames < t:
                    node_features_ = node_features[i:i + frames]
                    edge_features_ = edge_features[i:i + frames]
                    senders_ = senders[i:i + frames]
                    receivers_ = receivers[i:i + frames]
                    frame_ = {key: value[i:i + frames] for key, value in frame.items()}
                else:
                    node_features_ = node_features[i:t]
                    edge_features_ = edge_features[i:t]
                    senders_ = senders[i:t]
                    receivers_ = receivers[i:t]
                    frame_ = {key: value[i:t] for key, value in frame.items()}

                graph = MultiGraph(node_features_, edge_sets=[EdgeSet(edge_features_, senders_, receivers_)])

                output, target_normalized, acceleration = model(graph, frame_)
                loss = simulator_cloth.cloth_loss(output, target_normalized, frame_)

                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                loss_value.append(loss.numpy().item())
                loop.set_description(f'Step [{current_step}/{steps}]')
                loop.set_postfix(loss=loss.numpy().item())

            # print(f"step:{idx + e * train_len} => loss:{np.mean(np.array(loss_value)).item()}")
            train_summary_writer.add_scalar("train loss", np.mean(np.array(loss_value)).item(), e * train_len + idx)
            if flag:
                break

        opt_state = optimizer.state_dict()
        model_state = model.state_dict()
        state_dict = {"opt_state": opt_state, "model_state": model_state}
        paddle.save(state_dict, path=os.path.join(save_path, str(e + 1) + '.pdparams'))

        # perform validation
        # print(f"Epoch {e + 1} valid")
        # model.eval()
        # predicts = []
        # targets = []
        # loop = tqdm(valid_dataset, ncols=100)
        # for idx, item in enumerate(loop):
        #     node_features, edge_features, senders, receivers, frame = mesh_loader.data_to_feature(item)
        #     # split frame
        #     t, nodes, c = node_features.shape
        #     loss_value = []
        #     for i in range(0, t, frames):
        #         current_step = (e * train_len + idx) * t + i
        #         if current_step > steps:
        #             flag = 1
        #             break
        #
        #         if i + frames < t:
        #             node_features_ = node_features[i:i + frames]
        #             edge_features_ = edge_features[i:i + frames]
        #             senders_ = senders[i:i + frames]
        #             receivers_ = receivers[i:i + frames]
        #             frame_ = {key: value[i:i + frames] for key, value in frame.items()}
        #         else:
        #             node_features_ = node_features[i:t]
        #             edge_features_ = edge_features[i:t]
        #             senders_ = senders[i:t]
        #             receivers_ = receivers[i:t]
        #             frame_ = {key: value[i:t] for key, value in frame.items()}
        #
        #         graph = MultiGraph(node_features_, edge_sets=[EdgeSet(edge_features_, senders_, receivers_)])
        #         output, target_normalized, acceleration = model(graph, frame_)
        #         loss = cloth_loss(output, target_normalized, frame_)
        #         loop.set_postfix(loss=loss.numpy().item())
        #         predict = cloth_predict(acceleration, frame_)
        #         target = frame_['target|world_pos']
        #         predicts.append(paddle.to_tensor(predict.detach(), place=paddle.CPUPlace()).numpy())
        #         targets.append(paddle.to_tensor(target.detach(), place=paddle.CPUPlace()).numpy())
        #
        # mse = mean_squared_error(np.array(predicts), np.array(targets))
        # val_summary_writer.add_scalar("val mse", mse, step=e + 1)
        # print(f"epoch:{e + 1} => val mse:{mse}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", default=None, help="Path to checkpoint file used to resume training")
    parser.add_argument("--data_path", default=os.path.join(os.path.dirname(__file__), 'data', 'flag_simple'),
                        help="Path to dataset")
    parser.add_argument("--epoch", type=int, default=1000, help="Number of epochs to train (default :10M)")
    parser.add_argument("--save_path", default='./model_save/flag_simple', help="Path to save")
    parser.add_argument("--val_epoch", type=int, default=10, help="Every n epochs to valid (default :10)")
    parser.add_argument("--batch_size", type=int, default=10, help="Frames to load")

    args = parser.parse_args()
    train(data_path=args.data_path, val_epoch=args.val_epoch,
          save_path=args.save_path, checkpoint=args.checkpoint, frames=5)


#     train(data_path='data/data142705/',
#           epoch=1000, save_epoch=2, val_epoch=2, save_path='./model_save/flag_simple', checkpoint=None, bs=1)


if __name__ == '__main__':
    paddle.seed(123456)
    paddle.device.set_device("gpu:0")
    main()

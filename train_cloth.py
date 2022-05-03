import json
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
import common


def train(data_path='D:\Documents\Postgraduate\Code\MeshGraphNets\Data\\flag_simple',
          steps=10000000, save_path='./model_save/flag_simple',
          checkpoint=None, frames=1):
    train_dataset = mesh_loader.load_dataset(
        path=data_path,
        split='train',
        fields=['world_pos'],
        add_history=True,
        noise_scale=0.001,
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

    # 这里的指数学习率衰减与tf有不同
    lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=1e-4, milestones=[20, 23], gamma=0.1)
    optimizer = Adam(learning_rate=lr_scheduler, parameters=model.parameters())

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

    train_summary_writer = LogWriter(train_log_dir)

    # train
    flag = 0
    for e in range(26):
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

            train_summary_writer.add_scalar("train loss", np.mean(np.array(loss_value)).item(), e * train_len + idx)
            if flag:
                break
        lr_scheduler.step(e+1)
        opt_state = optimizer.state_dict()
        model_state = model.state_dict()
        state_dict = {"opt_state": opt_state, "model_state": model_state}
        paddle.save(state_dict, path=os.path.join(save_path, str(e + 1) + '.pdparams'))

        # perform validation
        with paddle.no_grad():
            model.eval()
            for idx, data in enumerate(tqdm(valid_dataset)):
                model.eval()

                node_features, edge_features, senders, receivers, frame = mesh_loader.data_to_feature(data)  # node_features: 399 x 1572 x 12
                frame_ = {key: value[0:1] for key, value in frame.items()}  # initial frame [1 x 1572 x 12]

                mask = paddle.equal(frame_['node_type'], common.NodeType.NORMAL)
                prev_pos = frame_['prev|world_pos']
                curr_pos = frame_['world_pos']
                trajectory = []
                rollout_loop = tqdm(range(399))

                # 进入滚动循环
                for t in rollout_loop:
                    node_features_, edge_features_, senders_, receivers_, frame_ = mesh_loader.data_to_feature(frame_)

                    graph = MultiGraph(node_features_, edge_sets=[EdgeSet(edge_features_, senders_, receivers_)])

                    # 计算输出
                    output, target_normalized, acceleration = model(graph, frame_)
                    next_pos = cloth_predict(acceleration, frame_)
                    next_pos = paddle.where(mask, next_pos, curr_pos)
                    trajectory.append(curr_pos)

                    # 更新位置信息到下一帧
                    prev_pos = curr_pos
                    curr_pos = next_pos
                    frame_['prev|world_pos'] = prev_pos
                    frame_['world_pos'] = curr_pos

                trajectory_predict = paddle.squeeze(paddle.stack(trajectory), axis=1).cpu()
                trajectory_ture = frame['world_pos'][0:399].cpu()

                error = paddle.mean(paddle.square(trajectory_predict - trajectory_ture), axis=-1)
                rmse_errors = {f'{horizon}_step_error': paddle.sqrt(paddle.mean(error[1:horizon + 1])).numpy()
                               for horizon in [1, 10, 20, 50, 100, 200, 398]}
                print(f'Epoch{e+1} RMSE Errors: {rmse_errors}')
                with open(os.path.join(save_path, 'eval_results.txt'), 'a+') as f:
                    f.write(f'Epoch{e+1}'+'\n')
                    f.write(json.dumps(rmse_errors))  # 加\n换行显示
                    f.write('\n')
                    f.close()


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint", "-c", default=None, help="Path to checkpoint file used to resume training")
    # parser.add_argument("--data_path", default=os.path.join(os.path.dirname(__file__), 'data', 'flag_simple'),
    #                     help="Path to dataset")
    # parser.add_argument("--epoch", type=int, default=26, help="Number of epochs to train (default :26epochs~10Msteps)")
    # parser.add_argument("--save_path", default='./model_save/flag_simple', help="Path to save")
    
    # args = parser.parse_args()
    # train(data_path=args.data_path, val_epoch=args.val_epoch,
    #       save_path=args.save_path, checkpoint=args.checkpoint, frames=5)


    train(data_path='D:\Documents\Postgraduate\Code\MeshGraphNets\Data\\flag_simple',
          save_path='./model_save/flag_simple', checkpoint=None)


if __name__ == '__main__':
    paddle.seed(123456)
    paddle.device.set_device("gpu:0")
    main()

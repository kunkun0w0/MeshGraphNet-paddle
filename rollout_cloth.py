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
from sklearn.metrics import mean_squared_error
import common


def to_numpy(t):
    """
    If t is a Tensor, convert it to a NumPy array; otherwise do nothing
    """
    try:
        return t.numpy()
    except:
        return t


@paddle.no_grad()
def rollout(checkpoint_path, dataset_path, batch_size=1, roll_steps=399):
    dataset = mesh_loader.load_dataset(
        path=dataset_path,
        split='test',
        fields=['world_pos'],
        add_history=True,
        noise_scale=0,
        noise_gamma=0)

    model = simulator_cloth.SimulatorCloth(
        node_input_size=12,
        edge_input_size=7,
        output_size=3,
        num_iterations=15,
        num_edge_types=1,
        hidden_size=128
    )

    state_dict = paddle.load(checkpoint_path)
    model_state = state_dict['model_state']
    model.set_state_dict(model_state)

    for idx, data in enumerate(dataset):
        model.eval()
        node_features, edge_features, senders, receivers, frame = mesh_loader.data_to_feature(data)  
        # node_features: 399 x 1572 x 12
        
        frame_ = {key: value[0:1] for key, value in frame.items()}  # initial frame [1 x 1572 x 12]
        mask = paddle.equal(frame_['node_type'], common.NodeType.NORMAL)
        prev_pos = frame_['prev|world_pos']
        curr_pos = frame_['world_pos']
        trajectory = []
        rollout_loop = tqdm(range(roll_steps))
        
        # 进入滚动循环
        for t in rollout_loop:
            node_features_, edge_features_, senders_, receivers_, frame_ = mesh_loader.data_to_feature(
                frame_)  # 将单帧数据转换（就是现在git上的那个meshloader里面的函数）

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

        trajectory_predict = paddle.squeeze(paddle.stack(trajectory), axis=1).cpu()  # roll_steps x points_num x 3
        trajectory_ture = frame['world_pos'][0:roll_steps].cpu()

        error = paddle.mean(paddle.square(trajectory_predict - trajectory_ture), axis=-1)
        rmse_errors = {f'{horizon}_step_error': paddle.sqrt(paddle.mean(error[1:horizon + 1])).numpy()
                       for horizon in [1, 10, 20, 50, 100, 200, 398]}
        result = {**frame, 'true_world_pos': trajectory_ture, 'pred_world_pos': trajectory_predict,
                  'errors': rmse_errors}

        result = {k: to_numpy(v) for k, v in result.items()}

        print(f'RMSE Errors: {rmse_errors}')

        with open(f'{idx:03d}.eval', 'wb') as f:
            pickle.dump(result, f)
        print(f'{idx} Evaluation results saved!')


rollout(checkpoint_path='model_save/3.pdparams', dataset_path='data/data142705/', roll_steps=399)





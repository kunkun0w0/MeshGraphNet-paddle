from model.simulator_cloth import SimulatorCloth, cloth_loss, cloth_predict
from model.model import MultiGraph, EdgeSet
from dataset import SimpleFlag
from paddle.io import DataLoader
from dataset.DataProcess import h5py_to_tensor
from sklearn.metrics import mean_squared_error
import paddle
from tqdm import tqdm
import numpy as np
import os
import pickle


def rollout(checkpoint_path, dataset_path, batch_size=1):
    dataset = SimpleFlag.simple_flag(
        dataset_dir=dataset_path,
        split='test',
        noise_scale=0,
        noise_gamma=0)

    model = SimulatorCloth.EncodeProcessDecode(
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
    model.eval()
    dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, drop_last=False)

    for epoch, item in enumerate(dataloader):
        node_features, edge_features, senders, receivers, frame = h5py_to_tensor(item)
        for _ in range(batch_size):
            predicteds = []
            targets = []
            tq = tqdm(range(node_features[batch_size].shape[0]))
            for i in tq:
                node_features_, edge_features_, senders_, receivers_ = \
                    node_features[_][i], edge_features[_][i], senders[_][i], receivers[_][i]
                frame_ = {key: value[_][i] for key, value in frame.items()}
                graph = MultiGraph(node_features_, edge_sets=[EdgeSet(edge_features_, senders_, receivers_)])
                output, target_normalized, acceleration = model(graph, frame_)
                loss = cloth_loss(output, target_normalized, frame_)
                tq.set_postfix('current loss: ', loss.numpy().item())
                predict = cloth_predict(acceleration, frame_)
                target = frame_['world_pos']
                predicteds.append(paddle.to_tensor(predict.detach(), place=paddle.CPUPlace()).numpy().item())
                targets.append(paddle.to_tensor(target.detach(), place=paddle.CPUPlace()).numpy().item())
            mse = mean_squared_error(np.array(predicteds), np.array(targets))
            print('epoch{}'.format(epoch+1), 'mse:', mse)
            result = [np.stack(predicteds), np.stack(targets)]
            os.makedirs('result', exist_ok=True)
            with open('result/result{}.pkl'.format(batch_size*epoch+_), 'wb') as f:
                pickle.dump(result, f)

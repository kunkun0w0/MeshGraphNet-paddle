from .model import EncodeProcessDecode
import model
from ..utils.normalization import Normalizer
import paddle.nn as nn
from ..common import NodeType
import paddle


class SimulatorCfd(nn.Layer):
    def __init__(self, node_input_size, edge_input_size, output_size, num_iterations, num_edge_types, hidden_size=128):
        super(SimulatorCfd).__init__()
        self.model = EncodeProcessDecode(node_input_size=node_input_size,
                                         edge_input_size=edge_input_size,
                                         output_size=output_size,
                                         num_iterations=num_iterations,
                                         num_edge_types=num_edge_types,
                                         hidden_size=hidden_size)
        # normalizer for the ground-truth acceleration
        self.output_normalizer = Normalizer(size=output_size, name='output_normalizer')
        # normalizer for the raw node features before the encoder MLP
        self.node_normalizer = Normalizer(size=node_input_size, name='node_normalizer')
        # normalizer for the raw edge features before the encoder MLP
        self.edge_normalizer = Normalizer(size=edge_input_size, name='edge_normalizer')

    def forward(self, graph, frame):
        # normalize node and edge features
        new_node_features = self.node_normalizer(graph.node_features)
        new_edge_sets = [graph.edge_sets[0]._replace(features=self.edge_normalizer(graph.edge_sets[0].features))]
        graph = model.MultiGraph(new_node_features, new_edge_sets)

        # pass through the encoder-processor-decoder architecture
        output = self.model(graph)

        # build target velocity change
        cur_velocity = frame['velocity']
        target_velocity = frame['target|velocity']
        target_velocity_change = target_velocity - cur_velocity
        target_normalized = self.output_normalizer(target_velocity_change)

        # for predict
        velocity_update = self.output_normalizer.inverse(output)

        return output, target_normalized, velocity_update


def cfd_loss(output, target_normalized, frame):
    # build loss
    loss_mask = paddle.cast(paddle.logical_or(paddle.equal(frame['node_type'][:, 0], NodeType.NORMAL),
                                              paddle.equal(frame['node_type'][:, 0], NodeType.OUTFLOW)),
                            dtype='float32')
    error = paddle.sum((target_normalized - output) ** 2, axis=1)
    loss = paddle.mean(error * loss_mask)
    return loss


def cfd_predict(velocity_update, frame):
    # integrate forward
    cur_velocity = frame['velocity']
    return cur_velocity + velocity_update

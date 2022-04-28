from model.model import EncodeProcessDecode
from model.model import MultiGraph
from utils.normalization import Normalizer
import paddle.nn as nn
from common import NodeType
import paddle


class SimulatorCloth(nn.Layer):
    def __init__(self, node_input_size, edge_input_size, output_size, num_iterations, num_edge_types, hidden_size=128):
        super(SimulatorCloth, self).__init__()
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
        graph = MultiGraph(new_node_features, new_edge_sets)

        # pass through the encoder-processor-decoder architecture
        output = self.model(graph)

        # build target acceleration
        cur_position = frame['world_pos']
        prev_position = frame['prev|world_pos']
        target_position = frame['target|world_pos']
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = self.output_normalizer(target_acceleration)

        # for predict
        acceleration = self.output_normalizer.inverse(output)

        return output, target_normalized, acceleration


def cloth_loss(output, target_normalized, frame):
    loss_mask = paddle.cast(paddle.equal(frame['node_type'][:, :, 0], NodeType.NORMAL), dtype='float32')
    error = paddle.sum(paddle.square(target_normalized - output), axis=2)
    loss = paddle.mean(error * loss_mask)
    return loss


def cloth_predict(acceleration, frame):
    # integrate forward
    cur_position = frame['world_pos']
    prev_position = frame['prev|world_pos']
    position = 2 * cur_position + acceleration - prev_position
    return position

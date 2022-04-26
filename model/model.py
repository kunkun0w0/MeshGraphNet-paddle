"""
Implements the core model common to MeshGraphNets
"""
import paddle
import paddle.nn as nn
import collections

EdgeSet = collections.namedtuple('EdgeSet', ['features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


# 注：这里仿照pytorch代码，不自定义感知机层数，全用4层
def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                           nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size))
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Layer):

    def __init__(self,
                 edge_input_size=7,
                 node_input_size=12,
                 hidden_size=128,
                 num_edge_types=2):
        super(Encoder, self).__init__()

        self.edge_encoder = [build_mlp(edge_input_size, hidden_size, hidden_size) for _ in range(num_edge_types)]  # edge_encoder
        self.node_encoder = build_mlp(node_input_size, hidden_size, hidden_size) # node_encoder

    def forward(self, graph):
        node_features = self.node_encoder(graph.node_features)
        edge_sets = []
        for i, edge_set in enumerate(graph.edge_sets):
            edge_features = self.edge_encoder[i](edge_set.features)
            edge_sets.append(edge_set._replace(features=edge_features))

        return MultiGraph(node_features, edge_sets)


class GraphNetBlock(nn.Layer):
    def __init__(self, hidden_size=128, num_edge_types=2):
        super(GraphNetBlock, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = (1+num_edge_types) * hidden_size

        # MLP for the node update rule
        self.node_update = build_mlp(nb_input_dim, hidden_size, hidden_size)
        # separate MLPs for the edge update rules
        self.edge_updates = [build_mlp(eb_input_dim, hidden_size, hidden_size) for _ in range(num_edge_types)]

    def _update_edge_features(self, node_features, edge_set, index):
        """
        Get the new edge features by aggregating the features of the
            two adjacent nodes
        :param node_features: Tensor with shape (n, d), where n is the number of nodes
                              and d is the number of input dims
        :param edge_set: EdgeSet; the edge set to update
        :param index: int; the index of the edge set in MultiGraph.edge_sets
        :return: Tensor with shape (m, d2), where m is the number of edges and
                 d2 is the number of output dims
        """
        sender_features = paddle.gather(node_features, edge_set.senders)
        receiver_features = paddle.gather(node_features, edge_set.receivers)
        features = [sender_features, receiver_features, edge_set.features]
        return self.edge_updates[index](paddle.concat(features, axis=-1))

    def _update_node_features(self, node_features, edge_sets):
        """
        Get the new node features by aggregating the features of all
            adjacent edges
        :param node_features: Tensor with shape (n, d), where n is the number of nodes
                              and d is the number of input dims
        :param edge_sets: list of EdgeSets; all edge sets in the graph
        :return: Tensor with shape (n, d2), where n is the number of nodes and d2
                 is the number of output dims
        """
        # num_nodes = tf.shape(node_features)[0]
        features = [node_features]
        for edge_set in edge_sets:
            # perform sum aggregation
            # 这里还有点问题, 应该用到num_nodes对segment_sum的结果维度做一个限制，paddle中没提供这个api
            features.append(paddle.incubate.segment_sum(edge_set.features, paddle.sort(edge_set.receivers)))
        return self.node_update(paddle.concat(features, axis=-1))

    def forward(self, graph, training=False):
        """
        Perform the message-passing on the graph
        :param graph: MultiGraph; the input graph
        :param training: unused
        :return: MultiGraph; the resulting graph after updating features
        """
        # update edge features
        new_edge_sets = []
        for i, edge_set in enumerate(graph.edge_sets):
            updated_features = self._update_edge_features(graph.node_features, edge_set, i)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # update node features
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets)

        # add residual connections
        new_node_features = new_node_features + graph.node_features
        new_edge_sets = [es._replace(features=es.features + old_es.features)
                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]

        return MultiGraph(new_node_features, new_edge_sets)


class Decoder(nn.Layer):
    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.node_features)


class EncodeProcessDecode(nn.Layer):
    # node_input_size=12 edge_input_size=7, output_size=2 num_edge_types
    def __init__(self, node_input_size, edge_input_size, output_size, num_iterations, num_edge_types, hidden_size=128):

        super(EncodeProcessDecode, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size,
                               hidden_size=hidden_size, num_edge_types=num_edge_types)
        self.num_iterations = num_iterations
        self.mp_blocks = []
        for _ in range(num_iterations):
            self.mp_blocks.append(GraphNetBlock(hidden_size=hidden_size, num_edge_types=num_edge_types))

        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size)

    def forward(self, graph, training=False):
        """
        Pass a graph through the model
        :param graph: MultiGraph; represents the mesh with raw node and edge features
        :param training: unused
        :return: Tensor with shape (n, d), where n is the number of nodes and d
                 is the number of output dims; represents the node update
        """
        graph = self.encoder(graph)
        for i in range(self.num_iterations):
            graph = self.mp_blocks[i](graph)
        return self.decoder(graph)




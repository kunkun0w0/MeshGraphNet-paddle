import paddle
import numpy as np
import common


def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arrange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arrange = x_arrange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arrange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


def h5py_to_tensor(data):
    velocity = data['velocity']
    velocity = paddle.to_tensor(np.array(velocity))
    node_type = paddle.to_tensor(np.squeeze(np.array(data['node_type'][:, 0])))
    node_type = paddle.nn.functional.one_hot(node_type, num_classes=common.NodeType.SIZE)
    node_type = paddle.expand(paddle.unsqueeze(node_type, axis=1),
                              shape=[node_type.shape[0], velocity.shape[1], node_type.shape[1]])

    node_features = paddle.concat([velocity, node_type], axis=-1)

    senders, receivers = common.triangles_to_edges(paddle.to_tensor(np.array(data['cells'])))

    world_pos = paddle.to_tensor(np.array(data['world_pos']))
    mesh_pos = paddle.to_tensor(np.array(data['mesh_pos']))

    w_shape = [senders.shape[0], senders.shape[1], 3]
    m_shape = [senders.shape[0], senders.shape[1], 2]

    relative_world_pos = paddle_gather(x=world_pos, index=senders.unsqueeze(axis=-1).expand(w_shape), dim=1) - \
                         paddle_gather(x=world_pos, index=receivers.unsqueeze(axis=-1).expand(w_shape), dim=1)
    relative_mesh_pos = paddle_gather(x=mesh_pos, index=senders.unsqueeze(axis=-1).expand(m_shape), dim=1) - \
                        paddle_gather(x=mesh_pos, index=receivers.unsqueeze(axis=-1).expand(m_shape), dim=1)

    edge_features = paddle.concat([
        relative_world_pos,
        paddle.norm(relative_world_pos, axis=-1, keepdim=True, p=2),
        relative_mesh_pos,
        paddle.norm(relative_mesh_pos, axis=-1, keepdim=True, p=2)], axis=-1)

    return node_features, edge_features, senders, receivers, data

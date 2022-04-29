import paddle
import common
from dataset import FlagSimple
from paddle.io import DataLoader


def load_dataset(path, split, fields, add_history, noise_scale=None, noise_gamma=None):
    print(f"Create {split} loader!")
    return FlagSimple.dataset(path=path, split=split, fields=fields, add_history=add_history,
                              noise_scale=noise_scale, noise_gamma=noise_gamma)


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


def data_to_feature(frame):
    velocity = frame['velocity']
    node_type = frame['node_type'].squeeze_(axis=-1)
    node_type = paddle.nn.functional.one_hot(node_type, num_classes=common.NodeType.SIZE)
    node_features = paddle.concat([velocity, node_type], axis=-1)

    senders, receivers = common.triangles_to_edges(frame['cells'])

    world_pos = frame['world_pos']
    mesh_pos = frame['mesh_pos']
    t, edges = senders.shape

    relative_world_pos = paddle_gather(x=world_pos, index=senders.unsqueeze(-1).expand([t, edges, 3]), dim=1) - \
                         paddle_gather(x=world_pos, index=receivers.unsqueeze(-1).expand([t, edges, 3]), dim=1)
    relative_mesh_pos = paddle_gather(x=mesh_pos, index=senders.unsqueeze(-1).expand([t, edges, 2]), dim=1) - \
                        paddle_gather(x=mesh_pos, index=receivers.unsqueeze(-1).expand([t, edges, 2]), dim=1)

    edge_features = paddle.concat([
        relative_world_pos,
        paddle.norm(relative_world_pos, axis=-1, keepdim=True, p=2),
        relative_mesh_pos,
        paddle.norm(relative_mesh_pos, axis=-1, keepdim=True, p=2)], axis=-1)

    return node_features, edge_features, senders, receivers, frame

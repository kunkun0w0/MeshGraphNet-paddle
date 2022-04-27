# https://github.com/jjdcorke/MeshGraphNets/blob/main/Models/common.py

import enum
import paddle


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def triangles_to_edges(faces):
    """
    Computes mesh edges from triangles.
    """
    # face -> [batch, frames, n_edge, 3]
    # flag_simple -> [batch, 399, 3028, 3]

    edges = paddle.concat([faces[..., :, 0:2], faces[..., :, 1:3],
                           paddle.stack([faces[..., :, 2], faces[..., :, 0]], axis=-1)], axis=-2)

    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers = paddle.min(edges, axis=-1)
    senders = paddle.max(edges, axis=-1)
    packed_edges = paddle.cast(paddle.stack([senders, receivers], axis=-1), paddle.int64)

    # remove duplicates and unpack
    unique_edges = paddle.cast(paddle.unique(packed_edges, axis=2), paddle.int32)
    senders, receivers = paddle.unstack(unique_edges, axis=-1)

    # create two-way connectivity
    return (paddle.concat([senders, receivers], axis=-1),
            paddle.concat([receivers, senders], axis=-1))

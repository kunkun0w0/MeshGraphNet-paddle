# MeshGraphNet-paddle
MeshGraphNet的paddle实现

id数据预处理：
```
python tfrecord.tools.tfrecord2idx ./data/flag_simple/train.tfrecord ./data/flag_simple/train.id
python tfrecord.tools.tfrecord2idx ./data/flag_simple/test.tfrecord ./data/flag_simple/test.id
python tfrecord.tools.tfrecord2idx ./data/flag_simple/valid.tfrecord ./data/flag_simple/valid.id
```

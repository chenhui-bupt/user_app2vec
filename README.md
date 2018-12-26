# user_app2vec
This project include network embedding and recommendation system.
## 1. network embedding
### 1.1 deepwalk
执行如下命令：
```
1. 生成邻接矩阵adj4deepwalk.mat
python 4deepwalk.py

2. 训练deepwalk模型，得到embedding
deepwalk --format mat --input ../graph_resources/adj4deepwalk.mat\
 --max-memory-data-size 0 --number-walks 10 \
 --representation-size 128 --walk-length 100 \
 --window-size 10 --workers 2 \
 --output ../embeddings_output/deepwalk.embeddings
```

### 1.2 hin2vec
更改代码：将main_py.py第99行的edge = ','.join([id2edge_class[int(id_)] for id_ in ids])
更改为：edge = ','.join([id2edge_class[int(id_)] for id_ in ids.split(',')])
然后执行命令：
```
python main_py.py res/karate_club_edges.txt node_vectors.txt metapath_vectors.txt -l 1000 -d 2 -w 2  
```  
本项目使用：
```
python main_py.py ../graph_resources/edges4hin2vec.txt node_vectors.txt metapath_vectors.txt -l 100 -k 10 -d 128 \
-w 4 -p 10 > ../logs/hin2vec.log 2>&1 &
```

## 2. cluster
对hin2vec生成metapath向量进行聚类，我们发现语义相同的元路径被聚到了一起。

| metapath |
| ------ |
| ['U-A', 'U-A,A-U,U-A'] | 
| ['A-U', 'A-U,U-A,A-U'] |
| ['A-U,U-A', 'A-U,U-A,A-U,U-A'] |
| ['U-A,A-U', 'U-A,A-U,U-A,A-U'] |
| ['C-A', 'C-A,A-U', 'C-A,A-U,U-A', 'A-C', 'A-C,C-A', ...] |
| ... |

## 3. recommendation system
推荐系统旨在对用户在每个时刻（小时）推荐App，整个过程包括数据处理，训练集的构建，dnn模型，训练预测，评估等。
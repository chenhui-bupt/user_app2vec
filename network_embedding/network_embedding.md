## 1. deepwalk
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

## 2. hin2vec
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




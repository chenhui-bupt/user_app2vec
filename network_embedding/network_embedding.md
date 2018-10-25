### hin2vec
更改代码：  
将main_py.py第99行的edge = ','.join([id2edge_class[int(id_)] for id_ in ids])
更改为：edge = ','.join([id2edge_class[int(id_)] for id_ in ids.split(',')])  
然后执行命令：  
python main_py.py res/karate_club_edges.txt node_vectors.txt metapath_vectors.txt -l 1000 -d 2 -w 2  
本项目使用：
```
python main_py.py ../graph_resources/edges4hin2vec.txt node_vectors.txt metapath_vectors.txt -l 100 -k 10 -d 128 -w 4 
> ../logs/hin2vec.log 2>&1 &
```


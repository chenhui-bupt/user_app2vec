# -*- coding:utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
import scipy.io as sio
import scipy.sparse as sparse
import findspark
findspark.init()
from pyspark import *
sc = SparkContext('local', 'deepwalk')


usage_file = '../resources/data0117'
usage_rdd = sc.textFile(usage_file).map(lambda x: x.split("|"))
usage_rdd = usage_rdd.filter(lambda x: x[2] != '柚宝宝孕育')
print(usage_rdd.count())

# 构建索引，对特征进行有序编码
userlist = usage_rdd.map(lambda x: x[0]).distinct().collect()
applist = usage_rdd.map(lambda x: x[2]).distinct().collect()
catlist = usage_rdd.map(lambda x: x[3]).distinct().collect()
userlist.sort()  # 有序
applist.sort()  # 有序
catlist.sort()  # 有序
nodelist = userlist + applist + catlist
node2id = dict(zip(nodelist, range(len(nodelist))))
print(len(userlist), len(applist), len(catlist), len(nodelist))
# 序列化保存编码
if not os.path.exists('../resources/node2id.pkl'):
	import pickle
	pickle.dump(node2id, open('../resources/node2id.pkl', 'wb'))

# edges
user_app = usage_rdd.map(lambda x: (node2id[x[0]], node2id[x[2]])).distinct()
app_cat = usage_rdd.map(lambda x: (node2id[x[2]], node2id[x[3]])).distinct()
edges = user_app.union(app_cat)
num_edges = edges.count()
print('node: %s, edges: %s' % (len(nodelist), num_edges))
# 稀疏邻接矩阵
row_ids = edges.map(lambda x: x[0]).collect()
col_ids = edges.map(lambda x: x[1]).collect()
data = [1] * len(row_ids)
adj_mat = sparse.coo_matrix((data, (row_ids, col_ids)), shape=(len(nodelist), len(nodelist)))
adj_mat = adj_mat + adj_mat.T  # 无向图邻接矩阵，要对称
sio.savemat('./graph_resources/adj4deepwalk.mat', {'network': adj_mat})
print('saved the sparse adjacency mat of the network successfully.')

if __name__ == '__main__':
	# test to check whether we have generated the graph data sucessfully or not.
	adj_mat = sio.loadmat('./graph_resources/adj4deepwalk.mat')['network']
	print(adj_mat.shape)
	assert adj_mat.shape == (len(nodelist), len(nodelist)), 'error in sparse adjacency matrix.'
	print(np.sum(adj_mat.todense()))
	assert np.sum(adj_mat.todense()) == (2 * num_edges), 'edges in sparse adjacency is not equal to origin data.'


# root = "../resources/data0117"
# data = pd.read_csv(root+"/data0117.csv", encoding='gbk')
# userlist = sorted(set(data['id']))
# applist = sorted(set(data['app']))
# nodelist = userlist + applist
# node2id = dict(zip(nodelist, range(len(nodelist))))
# id2node = dict(zip(range(len(nodelist)), nodelist))
# with open('./id2node.pkl', 'wb') as f:
# 	pickle.dump(id2node, f)
# with open('./node2id.pkl', 'wb') as f:
# 	pickle.dump(node2id, f)
# exit(0)
# groups = data.groupby(['id', 'app']).groups
# row_ids = []
# col_ids = []
# for g in groups:
# 	row_ids.append(node2id[g[0]])
# 	col_ids.append(node2id[g[1]])
# data = [1] * len(row_ids)
# user_app = sparse.coo_matrix((data, (row_ids, col_ids)), shape=(len(row_ids), len(col_ids)))
# user_app = user_app + user_app.T
# sio.savemat('./user_app0120.mat', {'network': user_app})








# -*- coding:utf-8 -*-
import pandas as pd
import pickle
import scipy.io as sio
import scipy.sparse as sparse

root = "~/works/datasets/nfp"
data = pd.read_csv(root+"/data0117.csv", encoding='gbk')
userlist = sorted(set(data['id']))
applist = sorted(set(data['app']))
nodelist = userlist + applist
node2id = dict(zip(nodelist, range(len(nodelist))))
id2node = dict(zip(range(len(nodelist)), nodelist))
with open('./id2node.pkl', 'wb') as f:
	pickle.dump(id2node, f)
with open('./node2id.pkl', 'wb') as f:
	pickle.dump(node2id, f)
exit(0)
groups = data.groupby(['id', 'app']).groups
row_ids = []
col_ids = []
for g in groups:
	row_ids.append(node2id[g[0]])
	col_ids.append(node2id[g[1]])
data = [1] * len(row_ids)
user_app = sparse.coo_matrix((data, (row_ids, col_ids)), shape=(len(row_ids), len(col_ids)))
user_app = user_app + user_app.T
sio.savemat('./user_app0120.mat', {'network': user_app})




# coding: utf-8
import os
import numpy as np
import pandas as pd

try:
    import findspark
except:
    os.system('pip install findspark')
    import findspark
findspark.init()
from pyspark import *
from pyspark import SparkContext as sc

# ---
# 1. 预处理
# ---
# '柚宝宝孕育'App数据有问题，应该删掉
# 
# ---

usage_file = '../resources/data0117'
usage_rdd = sc.textFile(usage_file).map(lambda x: x.split("|"))
usage_rdd = usage_rdd.filter(lambda x: x[2] != '柚宝宝孕育')
print(usage_rdd.take(5))
print(usage_rdd.count())


# ### 构建索引，对特征进行有序编码
userlist = usage_rdd.map(lambda x: x[0]).distinct().collect()
applist = usage_rdd.map(lambda x: x[2]).distinct().collect()
catlist = usage_rdd.map(lambda x: x[3]).distinct().collect()
userlist.sort()
applist.sort()
catlist.sort()
print(len(userlist), len(applist), len(catlist))
nodelist = userlist + applist + catlist
node2id = dict(zip(nodelist, range(len(nodelist))))
# 序列化保存编码
import pickle
pickle.dump(node2id, open('../resources/node2id.pkl', 'wb'))

rdd = usage_rdd.map(lambda x: (node2id[x[0]], x[2], x[3])).distinct()  # userid, app, cat
user_app = rdd.map(lambda x: '%s\tU\t%s\tA\tU-A' % (x[0], x[1])).distinct()
app_user = rdd.map(lambda x: '%s\tA\t%s\tU\tA-U' % (x[1], x[0])).distinct()
app_cat = rdd.map(lambda x: '%s\tA\t%s\tC\tA-C' % (x[1], x[2])).distinct()
cat_app = rdd.map(lambda x: '%s\tC\t%s\tA\tC-A' % (x[2], x[1])).distinct()
print(rdd.first())
print(user_app.first())
print(app_user.first())
print(app_cat.first())
print(cat_app.first())
print(user_app.count(), app_user.count(), app_cat.count(), cat_app.count())

out = user_app.union(app_user)
out = out.union(app_cat)
out = out.union(cat_app)
print(out.count())
out.repartition(1).saveAsTextFile('./graph_resources/edges4hin2vec')

# rename of edges4hin2vec/part-00000
os.system('mv graph_resources/edges4hin2vec/part-00000 graph_resources/edges4hin2vec.txt')


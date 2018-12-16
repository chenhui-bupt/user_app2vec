# coding: utf-8
import os
import sys
import random
import pandas as pd
import numpy as np
import pickle
import findspark
import collections
findspark.init()
from pyspark import *
sc = SparkContext.getOrCreate()


# TODO app2id和cat2id用来做onehot或者multihot

def load_data_to_df(file):
    # file: '../resources/data0117/'
    usage_rdd = sc.textFile(file).map(lambda x: x.split('|'))
    usage_rdd = usage_rdd.filter(lambda x: x[2] != '柚宝宝孕育')
    data = pd.DataFrame(usage_rdd.collect(), columns=['id','hour','app','app_cat','times'])  # convert to DataFrame
    # 确保数据类型
    data['hour'] = data['hour'].astype(int)
    data['times'] = data['times'].astype(int)
    print('the raw usage data have %s records' % len(data))
    print(data.dtypes)
    return data


def get_installed_apps(data):
    # 1. 用户的App安装列表
    installed_apps = data.groupby('id', as_index=False)['app'].agg({'installed_apps': lambda x: set(x.tolist())})
    return installed_apps


def get_current_apps(data):
    # 2. 用户当前时刻使用的App列表
    # 这一特征需要对数据进行按用户和时间进行排序，然后取时间偏置merge到一起
    positive_apps = data.groupby(['id', 'hour'], as_index=False)['app'].agg(
        {'positive_apps': lambda x: set(x.tolist())})
    return positive_apps


def get_app_attribute():
    # 3. App属性特征
    # 名称|类别|开发者|评分|下载量|描述
    app_cat = data.groupby('app', as_index=False).agg({'app_cat': lambda x: ','.join(set(x.tolist()))})
    print(len(app_cat))
    applist = list(app_cat['app'])
    appAttr = collections.defaultdict(list)
    with open('../resources/app_yingyongbao.txt', encoding='gbk') as f:
        for line in f:
            if '柚宝宝' in line:
                continue
            splits = line.split('|', maxsplit=5)
            if len(splits) == 6:
                appAttr['name'].append(splits[0])
                appAttr['cat'].append(splits[1])
                appAttr['developer'].append(splits[2])
                appAttr['ratings'].append(float(splits[3].split('=')[1]))
                appAttr['downloads'].append(float(splits[4].split('=')[1]))
                appAttr['description'].append(splits[5])
    appAttrDf = pd.DataFrame(appAttr)
    appAttrDf = pd.concat([app_cat, appAttrDf], axis=1)
    return appAttrDf


# 3. 负采样构建训练集
# 将用户当前时刻使用过app作为正例，对未使用过的app做负采样
# TODO 按照App的频率进行负采样，而不是各App之间等概采样
def neg_sampling(pos_samples, num_samples=5):
    num_samples = min(num_samples, len(applist) - len(pos_samples))
    neg_samples = []
    while len(neg_samples) < num_samples:
        sample = random.choice(applist)
        if sample in pos_samples or sample in neg_samples:
            continue
        neg_samples.append(sample)
    return neg_samples


def get_negative_apps(positive_apps, num_samples=5):
    negative_apps = positive_apps.copy()
    negative_apps['negative_apps'] = negative_apps['positive_apps'].apply(lambda x: set(neg_sampling(x, num_samples)))
    negative_apps.drop('positive_apps', axis=1, inplace=True)
    return negative_apps


# TODO 当前负样本是根据某一时刻的正样本产生的，如果当前时刻没有使用手机还要不要产生负样本？
# 4. dataframe文本内容列转行，一行转多行
'''python
df = pd.DataFrame({'A':['1','2','3'],'B':['1','2,3','4,5,6'],'C':['3','3','3']})
df = (df.set_index(['A','C'])['B']
       .str.split(',', expand=True)
       .stack()
       .reset_index(level=2, drop=True)
       .reset_index(name='B'))
print(df)
'''


def hadamard(vec1, vec2):
    return list(map(lambda x: x[0]*x[1], zip(vec1, vec2)))


def get_embeddings(file_name, id2node=None):
    embeddings = {}
    embeddings_file = os.path.join("../network_embedding/embeddings_output/", file_name)
    with open(embeddings_file, 'r') as f:
        for line in f:
            splits = line.split()
            if len(splits) == 2:
                print('node: %s, embedding_size: %s' % tuple(splits))
                continue
            if id2node:
                embeddings[id2node[int(splits[0])]] = list(map(float, splits[1:]))
            else:
                embeddings[splits[0]] = list(map(float, splits[1:]))
    return embeddings


def add_features(data):
    # 添加特征
    # 将之前生成用户App安装列表，上一时刻使用的App等特征扩展到数据集中
    # 1, 用户的app安装列表
    temp = installed_apps.copy()
    temp['installed_apps'] = temp['installed_apps'].apply(lambda x: len(x))
    data = pd.merge(data, temp, on=['id'], how='left')

    # 2, 上一时刻使用app列表，与原dataframe将来merge到一起
    last_apps = positive_apps.copy()
    last_apps['hour'] += 1  # 记录上一时刻使用的App列表
    last_apps.rename(columns={'positive_apps': 'last_apps'}, inplace=True)
    last_apps['last_apps'] = last_apps['last_apps'].apply(lambda x: len(x))
    data = pd.merge(data, last_apps, on=['id', 'hour'], how='left')
    data.fillna(0, inplace=True)  # fill na

    # 3, add app category feature and other App attribute information
    data = pd.merge(data, app_cat, on=['app'], how='left')

    # 4, 添加graph embedding特征
    # 该特征是通过对User-App网络进行网络表示学习得到节点的embedding表示，不仅包含低维稠密的信息，更包含节点之间的语义信息
    node2id = pickle.load(open('../resources/node2id.pkl', 'rb'))
    id2node = {v: k for k, v in node2id.items()}  # inversed index to node
    embeddings = get_embeddings('deepwalk.embeddings', id2node=id2node)
    user_apps = data.loc[:, ['id', 'app']].drop_duplicates()  # 去重避免重复计算
    vectors = list(map(lambda x: hadamard(embeddings[x[0]], embeddings[x[1]]), user_apps.values))
    vectors = pd.DataFrame(vectors, columns=['emb_%s' % i for i in range(128)])
    user_apps = pd.concat([user_apps, vectors], axis=1)
    data = pd.merge(data, user_apps, on=['id', 'app'], how='left')
    return data

if __name__ == '__main__':
    data = load_data_to_df('../resources/data0117')
    applist = list(set(data['app']))
    app_cat = data.groupby('app', as_index=False).agg({'app_cat': lambda x: ','.join(set(x.tolist()))})
    print(len(app_cat))
    installed_apps = get_installed_apps(data)
    positive_apps = get_current_apps(data)
    negative_apps = get_negative_apps(positive_apps)
    temp = positive_apps['positive_apps'].apply(lambda x: len(x))
    print('每个小时用户使用App的数量统计：')
    print(temp.describe())

    # 正例样本集
    pos = positive_apps.set_index(['id', 'hour'])['positive_apps'].apply(lambda x: ','.join(x)).str.split(',',\
            expand=True).stack().reset_index(level=2, drop=True).reset_index(name='app')
    pos['click'] = 1
    # 负例样本集
    neg = negative_apps.set_index(['id', 'hour'])['negative_apps'].apply(lambda x: ','.join(x)).str.split(',',\
            expand=True).stack().reset_index(level=2, drop=True).reset_index(name='app')
    neg['click'] = 0
    data = pd.concat([pos, neg], axis=0)
    print('pos: %s, neg: %s, total: %s' % (len(pos), len(neg), len(data)))
    data = add_features(data)
    data.to_csv('./data/deepwalk.csv', index=False)




# -*- coding:utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import random
import pickle
import findspark
import collections
findspark.init()
from pyspark import *
sc = SparkContext.getOrCreate()


def load_data_to_df(file):
    """
    从原始流量日志读取数据
    :param file: '../resources/data0117/'
    :return: dataframe
    """
    usage_rdd = sc.textFile(file).map(lambda x: x.split('|'))
    usage_rdd = usage_rdd.filter(lambda x: x[2] != '柚宝宝孕育')
    data = pd.DataFrame(usage_rdd.collect(), columns=['id','hour','app','app_cat','times'])  # convert to DataFrame
    data['hour'] = data['hour'].astype(int)  # 确保数据类型
    data['times'] = data['times'].astype(int)
    print('the raw usage data have %s records' % len(data))
    return data


def node2id(data, dump=False):
    """

    :param data:
    :return: node2id: {node: [id, type, count]}
    """
    type2node = {nodetype: list(data[nodetype]) for nodetype in ['id', 'app', 'app_cat']}
    node2id = {}
    i = 0
    for nodetype, nodelist in type2node.items():
        for node in nodelist:
            if node not in node2id:
                node2id[node] = [i, nodetype, 0]
                i += 1
            node2id[node][2] += 1
    assert i == len(node2id), 'your code is wrong!'
    if dump:
        pickle.dump(node2id, open('../resources/node2id.pkl', 'wb'))
        print("node2id has been dumped into '../resources/node2id.pkl'.")
    return node2id


def get_node2id(file):
    if not file:
        file = '../resources/node2id.pkl'
    node2id = pickle.load(open(file, 'rb'))
    print('load node2id from %s, it has %s nodes' % (file, len(node2id)))
    return node2id


def multihotEncoding(df, column):
    # TODO MultiHotEncoder
    mlb = MultiLabelBinarizer()
    X = df[column].values
    y = mlb.fit_transform(X)
    temp = pd.DataFrame(y, columns=list(map(lambda c: column + '_' + c, mlb.classes_)))
    df.drop(column, axis=1, inplace=True)
    df = pd.concat([df, temp], axis=1)
    return df


def to_categorical(y, num_class=None):  # onehot or multihot
    '''

    :param y: y.shape=(None,) or (None, None), 也就是onehot或者multihot
    :param num_class: 类别数
    :return:
    '''
    if not num_class:
        num_class = np.max(y) + 1
    out = np.zeros([len(y), num_class])
    for i in range(len(y)):
        out[i, y[i]] = 1
    return out


# TODO 用四天的数据来得到用户安装列表
def get_installed_apps(data):
    """
    用户的App安装列表, (id, {app1, app2, ...}), 然后进行multihot编码
    :param data: dataframe
    :return: dataframe
    """
    installed_apps = data.groupby('id', as_index=False)['app'].agg({'installed_apps': lambda x: set(x.tolist())})
    installed_apps = multihotEncoding(installed_apps, 'installed_apps')
    return installed_apps


def get_current_apps(data):
    """
    用户当前时刻使用的App列表, (id, {app1, app2, ...})
    :param data:
    :return:
    """
    positive_apps = data.groupby(['id', 'hour'], as_index=False)['app'].agg(
        {'positive_apps': lambda x: set(x.tolist())})
    return positive_apps


# TODO 按照App的频率进行负采样，而不是各App之间等概采样, 给定的applist不去重即可
def neg_sampling(pos_samples, applist, num_samples=5):
    """
    负采样构建训练集，将用户当前时刻使用过app作为正例，对未使用过的app做负采样
    :param pos_samples: set(apps), the apps are used in the current time
    :param applist: app list, not distinct
    :param num_samples: number of negative samples
    :return:
    """
    num_samples = min(num_samples, len(applist) - len(pos_samples))
    neg_samples = []
    while len(neg_samples) < num_samples:
        sample = random.choice(applist)
        if sample in pos_samples or sample in neg_samples:
            continue
        neg_samples.append(sample)
    return neg_samples


def get_negative_apps(positive_apps, applist, num_samples=5):
    """
    用户当前未使用的Apps集合，(id, {app1, app2, ...})
    :param positive_apps:
    :param applist:
    :param num_samples:
    :return:
    """
    negative_apps = positive_apps.copy()
    negative_apps['negative_apps'] = negative_apps['positive_apps'].apply(lambda x: set(neg_sampling(x, applist, num_samples)))
    negative_apps.drop('positive_apps', axis=1, inplace=True)
    return negative_apps


def get_app_attribute(data):
    """
    App属性特征, 名称|类别|开发者|评分|下载量|描述
    :param data:
    :return: dataframe with columns of app, app_cat, name, cat, developer, ratings, downloads, description
    """
    app_cat = data.groupby('app', as_index=False).agg({'app_cat': lambda x: ','.join(set(x.tolist()))})
    appAttr = collections.defaultdict(list)
    with open('../resources/app_yingyongbao.txt', encoding='gbk') as f:
        for line in f:
            if '柚宝宝' in line:  # 错误的数据
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


def generateTrainData(data):
    """
    输入的是原始数据，负采样构建负样本集, 然后求得特征，最后将特征一一添加到训练数据集上
    :param data:
    :return:
    """
    applist = list(data['app'])
    positive_apps = get_current_apps(data)
    negative_apps = get_negative_apps(positive_apps, applist)
    pos = positive_apps.set_index(['id', 'hour'])['positive_apps'].apply(lambda x: ','.join(x)).str.split(',', \
            expand=True).stack().reset_index(level=2, drop=True).reset_index(name='app')
    pos['click'] = 1
    neg = negative_apps.set_index(['id', 'hour'])['negative_apps'].apply(lambda x: ','.join(x)).str.split(',', \
            expand=True).stack().reset_index(level=2, drop=True).reset_index(name='app')
    neg['click'] = 0
    all_data = pd.concat([pos, neg], axis=0)
    print('pos: %s, neg: %s, total: %s' % (len(pos), len(neg), len(all_data)))

    # 1, installed apps of the user
    installed_apps = get_installed_apps(data)
    all_data = pd.merge(all_data, installed_apps, on=['id'], how='left')  # TrainData

    # 2, last hour used apps
    last_apps = positive_apps.copy()
    last_apps['hour'] += 1  # 记录上一时刻使用的App列表
    last_apps.rename(columns={'positive_apps': 'last_apps'}, inplace=True)
    last_apps = multihotEncoding(last_apps, 'last_apps')
    all_data = pd.merge(all_data, last_apps, on=['id', 'hour'], how='left')
    all_data.fillna(0, inplace=True)  # fill na

    # 3, add app category feature and other App attribute information
    app_attribute = get_app_attribute(data)
    app_attribute.drop(['name', 'cat', 'description'], axis=1, inplace=True)
    app_attribute = pd.get_dummies(app_attribute, columns=['app_cat', 'developer'])
    # TODO app_attribute get_dummies()
    all_data = pd.merge(all_data, app_attribute, on=['app'], how='left')

    # 4, graph embedding特征存放nodeid，留在tf.nn.lookup查询.

    # 把所有标称属性转换为id属性，并将一些特征onehot或multihot
    node2id = pickle.load(open('../resources/node2id.pkl', 'rb'))
    all_data['id'] = all_data['id'].apply(lambda x: node2id[x][0])
    all_data['app'] = all_data['app'].apply(lambda x: node2id[x][0])
    cols = list(all_data.columns)  # reorder the columns
    cols.pop(cols.index('id'))
    cols.pop(cols.index('app'))
    cols.pop(cols.index('click'))
    cols.insert(0, 'id')
    cols.insert(0, 'app')
    cols.append('click')
    all_data = all_data.loc[:, cols]
    print(all_data.head())
    print(list(all_data.columns))
    return all_data


if __name__ == '__main__':
    data = load_data_to_df('../resources/data0117')
    node2id(data, True)
    all_data = generateTrainData(data)
    # all_data.to_csv('./all_data.csv', index=False)

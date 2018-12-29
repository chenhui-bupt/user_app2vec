### 2. hin2vec代码梳理
1. 主类HIN()异构信息网络的类.

变量 | 类型 | 描述
---|---|---
self.node2id | dict | 所有节点倒排索引的字典, {node:id}
self.edge_class2id | dict | 边的类型的索引,{edge_class: id}
self.edge_class_id_available_node_class | dict | {edge_id:(from_class, to_class)}
self.class_nodes | dict | 某个类型节点的集合，{class: set([id1, id2])}
self.graph | dict | 图, {from_id: {to_id: {edge_id: weight}}}

其他重要变量:

类 | 变量 | 描述
---|---|---
HIN | edge_id | 每一个edge_class类型的边都有一个id，不是给每一条边一个id，是每一类型的边。
Path | path_id或path | 每个Path对象的基本属性，path是连续的边组成的元路径，中间用逗号隔开（所以不应叫path_id）。

2.2 network.py  
```python
class HIN():
    def __init__():
        self.graph = {} # {from_id: {to_id: {edge_class: weight}}}
        self.calss_nodes = {} # 异构网中同一类型的节点集合, {node_class: set([node_id])
        self.edge_class2id = {} # 边的类型倒排索引, {edge_class: edge_class_id}
        self.node2id = {} # 节点的倒排索引, {node: node_id}
        self.k_hop_neighbors = {} # 节点的k跳邻居集合，{k: {id_: set([to_ids])}}
        self.edge_class_id_available_node_class = {} # {edge_class_id: (from_node_class, to_node_class)
        
    def add_edge(self, from_node, from_class, to_node, to_class, edge_class, weight=1):
        """
        给图中添加一条有权边，并更新相关变量
        from_node: 源节点
        from_class: 源节点类型
        to_node: 漏节点
        to_class: 漏节点类型
        edge_class: 边的类型
        weight: 边的权值，default=1
        """
    
    def has_node(self, node):
        return node in self.node2id
    
    def has_edge(self, from_node, to_node, edge_class=None):
        """
        判断图中存不存在所指定的边
        """
    
    def print_statistics(self):
        for c, nodes in self.class_nodes.items():
            print c, len(nodes)  # 每个节点类别的节点数
        class_count = {}
        for class_edges in self.graph.values():
            for class_, to_ids in class_edges.items():
                if class_ not in class_count:
                    class_count[class_] = len(to_ids)
                    continue
                class_count[class_] += len(to_ids)
        print self.edge_class2id  # 边类别的索引
        for class_, count in class_count.items():
            print class_, count  # 每个边类别的节点数
    
    def a_random_walk(self, node, length):
        """
        以node节点为起点的一条长度为length的随机游走路径
        """
    
    def create_node_choices(self):
        """
        self.node_choices: {from_id: [(to_id, edge_class_id)]}, 并对（to_id,edge_class_id)复制int(weight)倍
        return: walk: [<node_id>, <edge_class_id>, <node_id>, ...]
        """
    
    def random_walks(self, count, length):
        """
        每个节点为起点的count条长度为length的随机游走路径
        """
    
    def random_select_edges_by_classes(self, from_class, to_class, edge_class, count):
        """
        按照指定节点类型，边的类型，采样count条边
        """
    def in_k_hop_neighborhood(self, id1, id2, k):
        """
        判断id2是不是id1的第k跳邻居
        """
    def _get_k_hop_neighborhood(self, id_, k):
        """
        得到节点id_的k跳路径上的所有的邻居集合
        """
    
    def get_candidates(self, node_id, k, to_node_class):
        """
        节点node_id的k跳邻居，去除直接邻居中，节点类型为to_node_class的节点集合
        """
    
    def get_shortest_distance(self, node_id1, node_id2, max_=None):
        """
        宽度优先搜索，计算两个节点之间的最短路径
        """
    
    def random_remove_edges(self, edge_class, ratio=0.5, seed=None):
        """
        随机丢弃指定类型指定比例的边，保证边的类别的均衡
        """
    
    def random_select_neg_edges(self, edge_class, count, seed=None):
        """
        随机选择count条egde_class类型的负样本边
        edge_class: 边的类型，即关系
        count: 负样本数量
        return: selected: [(from_id, to_id), ...]
        """
    
    def generate_test_set(self, path, count, seed=None):
        """
        重点函数：产生测试集，针对指定的一条元路径，产生count条相同起止节点类型(from_class, to_class)的节点对，如果两点之间有path这样的元路径，则label=1，否则label=0
        用到函数：_check_path()
        path: list(edge_class_id), 元路径
        count: 数量
        """
    def _check_path(self, from_id, to_id, path):
        """
        from_id: 源节点
        to_id: 漏节点
        path: 元路径，list(edge_class_id)
        """
```
2.3 mp2vec_s.py
```python
class Common(object):
    def __init__(self):
        self.node_vocab = None
        self.path_vocab = None
        self.node2vec = None
        self.path2vec = None
    
    def train(self, training_fname, seed=None):
        raise NotImplementedError
    
    def dump_to_file(self, output_name, type='node'):
        """
        输出节点的node_id和vec向量
        """

class MP2Vec(Common):
    def __init__(self, size=100, window=10, neg=5,
                   alpha=0.005, num_processes=1, iterations=1,
                   normed=True, same_w=False,
                   is_no_circle_path=False):
        '''
            size:      Dimensionality of word embeddings
            window:    Max window length
            neg:       Number of negative examples (>0) for
                       negative sampling, 0 for hierarchical softmax
            alpha:     Starting learning rate
            num_processes: Number of processes
            iterations: Number of iterations
            normed:    To normalize the final vectors or not
            same_w:    Same matrix for nodes and context nodes
            is_no_circle_path: Generate training data without circle in the path
        '''
    def train(self, g, training_fname, class2node_ids,
                                       seed=None,
                                       edge_class_inverse_mapping=None,
                                       k_hop_neighbors=None,
                                       id2vec_fname=None,
                                       path2vec_fname=None):
        """
        g: graph, HIN
        training_fname: 训练集文件，每行的内容是一条随机游走路径，node_id, edge_id, node_id, ...
        class2node_ids:
        id2vec_fname: 预训练的节点向量的文件名
        path2vec_fname: 预训练的边向量的文件名
        """
    
    def init_net(dim, node_size, path_size,
                 id2vec=None, path2vec=None):
        '''
        dim: embedding size
        node_size: 节点的数量
        path_size: 边/关系的数量
        id2vec: 初始的节点向量
        path2vec: 初始的边/关系向量
        return
            Wx: a |V|*d matrix for input layer to hidden layer，节点x从onehot映射到d的向量空间上
            Wy: a |V|*d matrix for hidden layer to output layer，节点y从onehot映射到d的向量空间上
            Wpath: a |paths|*d matrix for hidden layer to output layer，边/关系从onehot映射到d的空间上
        '''

class UnigramTable(object):
    '''
        For negative sampling.
        A list of indices of words in the vocab
        following a power law distribution.
    '''
    def __init__(self, g, vocab, seed=None, size=1000000, times=1, node_ids=None, uniform=False):
    
    def generate_table(g, vocab, table_size, node_ids, uniform):
        """
        计算按词频分布的词表，词的频率越高，其id在table表中出现的次数越高，不过这个频率要0.75次方，采样
        """

def get_context(node_index_walk, edge_walk, walk, path_vocab,
                index, window_size, no_circle=False):
    """
    随机游走路径中某个位置节点的上下文
    """
    start = max(index - window_size, 0)
    end = min(index + window_size + 1, len(node_index_walk))
    
def train_process(pid, node_vocab, path_vocab, Wx, Wy, Wpath,
                  tables,
                  neg, starting_alpha, win, counter,
                  iterations, training_fname, start_end,
                  same_w, k_hop_neighbors,
                  is_no_circle_path):
    """
    pid: 进程号
    node_vocab: 节点的词表，元素对象是节点id，数量等属性
    path_vocab: 边/关系的词表，元素是边的id
    Wx: 节点的向量表示
    Wy: 节点的向量表示
    Wpath: 边/关系的向量表示
    tables: 词表，表征节点在词表中出现次数的分布
    neg: 负采样样本数
    start_alpha: 学习率
    win：
    counter：多进程全局变量，计数器
    iteratiions：迭代次数
    training_fname：训练“语料”
    start_end:
    same_w:
    k_hop_neighbors: k跳路径上的所有邻居
    is_no_circle_path: 路径中有没有环
    """
    
    def dev_sigmoid(x):
        """
        [s(x)]' = s(x) * (1 - s(x)), sigmoid函数的导函数
        """
    
    def get_wp2_wp3(wp):
        """
        wp2: relu激活函数的导数,{0, x<= 0; 1, x>0}
        wp3: sigmoid激活函数的导数，(-6, 6)有导数，之外梯度近似为0
        """
    
```
2.4 ds/mp.py
这个类主要是封装了hin的基本元素节点Node和路径Path的属性类

```python
class Node(object):
    def __init__(self, node_id, count=0):
        self.node_id = node_id
        self.count = count
    
class NodeVocab(object):
    """
    节点的词表，存放节点对象，词表的倒排索引、其他属性等
    """
    def __init__(self):
        self.nodes = []
        self.node2index = {}
        self.node_count = 0
    
    def add_node(self, node_id):
        ''
    @staticmethod
    def load_from_network(g):
    
    @staticmethod
    def load_from_file(fname, available_ids=None):
        '''
        input:
            training_fname:
                each line: <node_id> <edge_id> ...
            available_ids: set([<node_id>])
                node_id should be a string，只取指定的node_id
        '''

class Path(object):
    '''
        Path for PathVocab
    '''
    def __init__(self, path_id, count=0, is_inverse=False):
        self.path_id = path_id
        self.count = count
        self.is_inverse = is_inverse

class PathVocab(object):
    '''
        a path is a list of edges
    '''
    def __init__(self):
        self.paths = []
        self.path2index = {}
        self.path_count = 0

    def add_path(self, path_id):
        ''
    def load_from_file(fname, window_size, inverse_mapping=None):
        '''
            input:
                training_fname:
                    each line: <node_id> <edge_id> ...
                window_size: the maximal window size for paths,
                元路径的窗口大小，比如U-A,A-C,C-A,A-U
        '''
        with open(fname, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                tokens = [t for i, t in enumerate(tokens) if i % 2 == 1]
                for w in range(window_size):
                    for i in range(len(tokens)-w):
                        edges = tokens[i:i+1+w]
                        path = ','.join(edges)
    
    

```
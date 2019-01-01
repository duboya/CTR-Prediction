import gc
import numpy as np
import pandas as pd
import tensorflow as tf

##################################
# 0. Functions
##################################
class Config(object):
    """
    用来存储一些配置信息
    """
    self.feature_dict = None
    self.feature_size = None
    self.field_size = None
    self.embedding_size = 8

    self.epochs = 20
    self.deep_layers_activation = tf.nn.relu

    self.loss = "logloss"
    self.l2_reg = 0.1
    self.learning_rate = 0.1

def FeatureDictionary(dfTrain=None, dfTest=None, numeric_cols=None, ignore_cols=None):
    """
    目的是给每一个特征维度都进行编号。
    1. 对于离散特征，one-hot之后每一列都是一个新的特征维度。所以，原来的一维度对应的是很多维度，编号也是不同的。
    2. 对于连续特征，原来的一维特征依旧是一维特征。
    返回一个feat_dict，用于根据 原特征名称和特征取值 快速查询出 对应的特征编号。
    :param dfTrain: 原始训练集
    :param dfTest:  原始测试集
    :param numeric_cols: 所有数值型特征
    :param ignore_cols:  所有忽略的特征. 除了数值型和忽略的，剩下的全部认为是离散型
    :return: feat_dict, feat_size
             1. feat_size: one-hot之后总的特征维度。
             2. feat_dict是一个{}， key是特征string的col_name, value可能是编号（int），可能也是一个字典。
             如果原特征是连续特征： value就是int，表示对应的特征编号；
             如果原特征是离散特征：value就是dict，里面是根据离散特征的 实际取值 查询 该维度的特征编号。 因为离散特征one-hot之后，一个取值就是一个维度，
             而一个维度就对应一个编号。
    """
    assert not (dfTrain is None), "train dataset is not set"
    assert not (dfTest is None), "test dataset is not set"

    # 编号肯定是要train test一起编号的
    df = pd.concat([dfTrain, dfTest], axis=0)

    # 返回值
    feat_dict = {}

    # 目前为止的下一个编号
    total_cnt = 0

    for col in df.colums:
        if col in ignore_cols:    # 忽略的特征不参与编号
            continue

        # 连续特征只有一个编号
        if col in numeric_cols:
            feat_dict[col] = total_cnt
            total_cnt += 1
            continue

        # 离散特征，有多少取值就有多少个编号
        unique_vals = df[col].unique()
        unique_cnt = df[col].nunique()
        feat_dict[col] = dict(zip(unique_vals, range(total_cnt, total_cnt + unique_cnt)))
        total_cnt += unique_cnt

    feat_size = total_cnt
    return feat_dict, feat_size

def parse(feat_dict=None, df=None, has_table=False):
    """
   构造FeatureDict，用于后面Embedding
   :param feat_dict: FeatureDictionary生成的。用于根据col和value查询出特征编号的字典
   :param df: 数据输入。可以是train也可以是test,不用拼接
   :param has_label:  数据中是否包含label
   :return:  Xi, Xv, y
   """
    assert not (df is None), "df is not set"

    dfi = df.copy()

    if has_table:
        y = df['target'].values.tolist()
        dfi.drop(['id', 'target'], axis=1, inplace=True)
    else:
        ids = dfi['id'].values.tolist()    # 预测样本的ids
        dfi.drop(['ids'], axis=1, inplace=True)    # axis : {0 or ‘index’, 1 or ‘columns’}, default 0

    # dfi是Feature index,大小和dfTrain相同，但是里面的值都是特征对应的编号。
    # dfv是Feature value, 可以是binary(0或1), 也可以是实值float，比如3.14
    dfv = dfi.copy()

    for col in dfi.columns:
        if col in IGNORE_FEATURES:    # 用到的全局变量： IGNORE_FEATURES, NUMERIC_FEATURES
            dfi.drop([col], axis=1, inplace=True)
            dfv.drop([col], axis=1, inplace=True)
            continue

        if col in NUMERIC_FEATURES:    # 连续特征1个维度，对应1个编号，这个编号是一个定值
            dfi[col] = feat_dict[col]
        else:
            # 离散特征。不同取值对应不同的特征维度，编号也是不同的。
            dfi[col] = dfi[col].map(feat_dict[col])
            dfv[col] = 1.0

# —*- coding: utf-8 -*-


class Config(object):
    def __init__(self, word_embedding_dimension=300,
                 epoch=3, learning_rate=0.001, batch_size=50):
        self.word_embedding_dimension = word_embedding_dimension     # 词向量的维度
        self.epoch = epoch                                           # 遍历样本次数
        self.lr = learning_rate
        self.batch_size = batch_size
        self.cuda = False
        self.max_sent_length = 300
        self.max_label_length = 3
        self.hidden_linear_size = 200
        self.max_k = 10


# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from torch.utils import data
import random
import pickle
from metadata_preprocessing import read_data, data_clean, data_analysis
from config import Config
config = Config()


random.seed(1024)


USE_CUDA = torch.cuda.is_available()
gpus = [0]
# torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))


def get_2idx(vocab, tag_set):
    word2index = {'<PAD>': 0, '<UNK>': 1}
    for w in vocab:
        if w not in word2index:
            word2index[w] = len(word2index)

    index2word = {v:k for k,v in word2index.items()}

    tag2index = {'None': 0}
    for t in tag_set:
        if t not in tag2index:
            tag2index[t] = len(tag2index)

    index2tag = {v:k for k,v in tag2index.items()}

    return word2index, index2word, tag2index, index2tag


def prepare_data(train_data, word2index, tag2index):
    X_p, y_p = [], []
    for pair in zip(train_data['all_text'], train_data['tag']):
        X_p.append(prepare_sequence(pair[0], word2index).view(1, -1))
        # y_p.append(Variable(LongTensor([tag2index[pair[1]]])).view(1, -1))
        y_p.append(prepare_sequence(pair[1], tag2index).view(1, -1))

    data_pair = list(zip(X_p, y_p))
    random.shuffle(data_pair)

    train_data = data_pair[: int(len(data_pair) * 0.9)]
    test_data = data_pair[int(len(data_pair) * 0.9):]

    return train_data, test_data


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

# 可以用tensorflow的padding函数
def pad_to_batch(batch, word2index, tag2index):
    x, y = zip(*batch)
    # max_x = max([s.size(1) for s in x])

    x_p = []
    y_p = []
    for i in range(len(batch)):

        if x[i].size(1) < config.max_sent_length:
            x_p.append(torch.cat([x[i], Variable(LongTensor([word2index['<PAD>']] * (config.max_sent_length - x[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(x[i][:, :config.max_sent_length])

        # print(y[i], y[i].data)
        if y[i].size(1) < config.max_label_length:
            # y_p.append(torch.cat([y[i], Variable(LongTensor([tag2index['None']] * (config.max_label_length - y[i].size(1)))).view(1, -1)], 1))
            padded_y = torch.cat([y[i], Variable(LongTensor([tag2index['None']] * (config.max_label_length - y[i].size(1)))).view(1, -1)], 1)
            label_tensor = torch.zeros(len(tag2index)).scatter_(0, padded_y.squeeze(0), 1).long()
            y_p.append(label_tensor.view(1, -1))
            # print(padded_y.data, label_tensor.data)
        else:
            # y_p.append(y[i])
            label_tensor = torch.zeros(len(tag2index)).scatter_(0, y[i].squeeze(0), 1).long()
            y_p.append(label_tensor.view(1, -1))


    return torch.cat(x_p), torch.cat(y_p)


# word2index, index2word, tag2index, index2tag = get_2idx(vocab, tag_set)
# train_data, test_data = prepare_data(train_data, word2index, tag2index)
#
# print('size of train_data', len(train_data))
# # print(train_data[0])


if __name__ == '__main__':
    # read data and pre-processing
    all_data = read_data()
    all_data = data_clean(all_data)
    vocab, tag_set = data_analysis(all_data)

    # prepare data
    word2index, index2word, tag2index, index2tag = get_2idx(vocab, tag_set)
    train_data, test_data = prepare_data(all_data, word2index, tag2index)

    # for fast testing
    # train_data = train_data[:200]
    # test_data = test_data[:20]

    print('size of train_data', len(train_data))

    # save data
    print('saving data……')
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/train_data.pkl', 'wb') as fp:
        pickle.dump(train_data, fp)
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/test_data.pkl', 'wb') as fp:
        pickle.dump(test_data, fp)
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/word2index.pkl', 'wb') as fp:
        pickle.dump(word2index, fp)
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/tag2index.pkl', 'wb') as fp:
        pickle.dump(tag2index, fp)
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/index2tag.pkl', 'wb') as fp:
        pickle.dump(index2tag, fp)
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/index2word.pkl', 'wb') as fp:
        pickle.dump(index2word, fp)
#coding:utf8
import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle
import math
from config import Config
from model import TextCNN
from collections import Counter
from data_preparation import getBatch, pad_to_batch


def val_loss(model, dataset):
    loss_f = nn.MultiLabelSoftMarginLoss()
    print('\ncomputing the loss on testing set……')

    loss_sum = 0
    count = 0
    # get the loss on the test data
    for i, batch in enumerate(getBatch(config.batch_size, dataset)):
        data, label = pad_to_batch(batch, word2index, tag2index)

        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            labels = label.cuda()

        score = model(data, True)
        # print(torch.topk(score, 3, dim=1)[1])
        loss = loss_f(score, label.float())


        loss_sum += loss.data
        count += 1
    print('loss on testing set:', loss_sum/count)

    del score


    return loss_sum/count

def get_score(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)
    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0]  # 在各个位置上总命中数量
    total_label_at_pos_num = [0, 0, 0]
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += len(set(predict_labels)) # |P|
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set) # |T|
        for pos, label in zip(range(0, min(len(predict_labels), 3)), predict_labels):
            total_label_at_pos_num[pos] += 1
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    proportion = 3
    for pos, right_num in zip(range(3), right_label_at_pos_num):
        # precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
        precision += (right_num / float(total_label_at_pos_num[pos])) * (float(proportion)/5)
        proportion -= 1

    # precision = float(right_label_num / sample_num)
    recall = float(right_label_num) / all_marked_label_num

    return 2*(precision * recall) / (precision + recall + 0.0000000000001), precision, recall, right_label_at_pos_num



if __name__ == '__main__':

    # load data and model
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/train_data.pkl', 'rb') as fp:
        train_data = pickle.load(fp)
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/test_data.pkl', 'rb') as fp:
        test_data = pickle.load(fp)
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/word2index.pkl', 'rb') as fp:
        word2index = pickle.load(fp)
    with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/tag2index.pkl', 'rb') as fp:
        tag2index = pickle.load(fp)

    # build model
    config = Config()
    model = TextCNN(len(word2index), config.word_embedding_dimension, len(tag2index))
    model.load_state_dict(torch.load('save_model.pth'))
    model.eval()

    # test
    print('start testing……')
    result = []
    # for test in train_data:
    for i, batch in enumerate(getBatch(config.batch_size, test_data)):
        data, label = pad_to_batch(batch, word2index, tag2index)

        with torch.no_grad():
            score = model(data)
        pred = torch.topk(score, 3, dim=1)[1].data.tolist()
        # print('pred:', pred)

        target = torch.topk(label, 3, dim=1)
        # print('target:', target, target[1], target[0])

        true_index = target[1] # 索引,即位置
        true_label = target[0] # 值(是0或1)

        tmp = []
        for jj in range(label.size(0)): # batch size
            true_index_ = true_index[jj] # index and value of each instance
            true_label_ = true_label[jj]
            true = true_index_[true_label_ > 0] # only keep indices for whose value is 1 not 0
            tmp.append((pred[jj], true.tolist()))
        # print(tmp) # length of pred is always 3, but length for some true is 2

        result.extend(tmp)

        del score
        # break

    f_score, prec_, recall_, _ss = get_score(result)
    print('\nf_score: {f_score}\n precision: {prec}\n recall: {recall}\n the number of right label at each position: {num_pos}'.format(f_score=f_score, prec=prec_, recall=recall_, num_pos=_ss))

    val_loss(model, test_data)





# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from config import Config
from model import TextCNN
import random
import gensim
import numpy as np
import pickle
from data_preparation import getBatch, pad_to_batch

torch.manual_seed(1)
random.seed(1024)

def val(model, dataset):
    loss_f = nn.MultiLabelSoftMarginLoss()
    model.eval()
    print('computing the loss on validation set……')

    loss_sum = 0
    count = 0
    # get the loss on the test data
    for i, batch in enumerate(getBatch(config.batch_size, dataset)):
        data, label = pad_to_batch(batch, word2index, tag2index)

        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            labels = label.cuda()

        score = model(data, True)
        loss = loss_f(score, label.float())

        loss_sum += loss.data
        count += 1
    print('loss on validation set:', loss_sum/count)

    del score

    model.train()

    return loss_sum/count

# # read data and pre-processing
# all_data = read_data()
# all_data = data_clean(all_data)
# vocab, tag_set = data_analysis(all_data)
#
#
# # prepare data
# word2index, index2word, tag2index, index2tag = get_2idx(vocab, tag_set)
# train_data, test_data = prepare_data(all_data, word2index, tag2index)
#
# # for fast testing
# # train_data = train_data[:200]
# # test_data = test_data[:20]
#
# print('size of train_data', len(train_data))
#
# # save data
# print('saving data……')
# with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/train_data.pkl', 'wb') as fp:
#     pickle.dump(train_data, fp)
# with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/test_data.pkl', 'wb') as fp:
#     pickle.dump(test_data, fp)
# with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/word2index.pkl', 'wb') as fp:
#     pickle.dump(word2index, fp)
# with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/tag2index.pkl', 'wb') as fp:
#     pickle.dump(tag2index, fp)

# load data
print('loading data……')
with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/train_data.pkl', 'rb') as fp:
    train_data = pickle.load(fp)
with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/test_data.pkl', 'rb') as fp:
    test_data = pickle.load(fp)
with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/word2index.pkl', 'rb') as fp:
    word2index = pickle.load(fp)
with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/tag2index.pkl', 'rb') as fp:
    tag2index = pickle.load(fp)
with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/index2word.pkl', 'rb') as fp:
    index2word = pickle.load(fp)
with open('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/index2tag.pkl', 'rb') as fp:
    index2tag = pickle.load(fp)




# load pre-trained word vectors
print('loading word vectors……')
word_v = gensim.models.KeyedVectors.load_word2vec_format('/Users/pengyiliu/study/GoogleNews-vectors-negative300.bin', binary=True)
pretrained = []

for key in word2index.keys():
    try:
        pretrained.append(word_v[word2index[key]])
    except:
        pretrained.append(np.random.randn(300))

pretrained_vectors = np.vstack(pretrained)


# set parameters
if torch.cuda.is_available():
    torch.cuda.set_device(0)
config = Config()


# build model
model = TextCNN(len(index2word), config.word_embedding_dimension, len(index2tag))
# embeds = nn.Embedding(config.word_num, config.word_embedding_dimension)
model.init_weights(pretrained_vectors) # initialize embedding matrix using pretrained vectors

if torch.cuda.is_available():
    model.cuda()
    # embeds = embeds.cuda()

# criterion = nn.CrossEntropyLoss()
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

count = 0
loss_sum = 0
# Train the model
print('start training……')
for epoch in range(config.epoch):
    for i, batch in enumerate(getBatch(config.batch_size, train_data)):
        data, label = pad_to_batch(batch, word2index, tag2index)

        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            labels = label.cuda()

        model.zero_grad()
        # input_data = embeds(autograd.Variable(data))
        out = model(data, True)
        loss = criterion(out, label.float())

        loss_sum += loss.data
        count += 1

        if count % 100 == 0:
            print("epoch", epoch, end='  ')
            print("The loss is: %.5f" % (loss_sum/100))

            test_loss = val(model, test_data)

            loss_sum = 0
            count = 0

        loss.backward()
        optimizer.step()
    # save the model in every epoch
    # model.save('checkpoints/epoch{}.ckpt'.format(epoch))

# save model
print('saving model……')
torch.save(model.state_dict(), 'save_model.pth')

# print('start testing……')
# accuracy = 0
# for test in test_data:
#     pred = model(test[0]).max(1)[1]
#     pred = pred.data.tolist()[0]
#     target = test[1].data.tolist()[0][0]
#     if pred == target:
#         accuracy += 1
#     # else:
#     #     print(pred, target)
#
# print(accuracy/len(test_data) * 100)




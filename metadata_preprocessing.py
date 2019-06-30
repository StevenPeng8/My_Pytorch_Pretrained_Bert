import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import re
import math
from config import Config
flatten = lambda l: [item for sublist in l for item in sublist]
config = Config()

def read_data():
    df = pd.read_json('/Users/pengyiliu/Desktop/UoS/Dissertation_Project/Implementation/arxivData.json')
    print(df.shape)

    # get summary and tag
    df["all_text"] = df["title"] + ". " + df["summary"]
    # df["all_text"] = df["title"]
    df = df.loc[:, ["all_text", "tag"]]
    print(df.head(8))

    # extract multiple tags for each instance
    df['tag'] =df['tag'].map(lambda x: str(x).split('{\'term\':')[1:])

    for i in range(df['tag'].size):
        new_tag = []
        for tag in df['tag'].iloc[i]:
            new = tag.split(',')[0][2:-1]
            pattern = re.compile('[0-9]+')
            match = pattern.findall(new)
            if not match:
                new_tag.append(new)

        df['tag'].iloc[i] = new_tag

    print(df.head(8))

    return df

def data_clean(train_data):
    # get rid of '\n'
    train_data['all_text'] = train_data['all_text'].map(lambda x: x.replace('\n', ' '))
    # remove non_letters
    train_data['all_text'] = train_data['all_text'].map(lambda x: re.sub("[^a-zA-Z]", " ", x))
    # lower and split
    train_data['all_text'] = train_data['all_text'].map(lambda x: x.lower().split(' '))

    # get rid of some meaningless summary
    idx_list = []
    idx = 0
    for doc in train_data['all_text']:
        if len(doc) <= 50:
            idx_list.append(idx)
        idx += 1

    # exclude some classes whose frequencies are low
    distribution = Counter(flatten(train_data['tag']))
    print(distribution)

    idx_tag_1 = 0
    for doc in train_data['tag']:
        new = []
        for tag in doc:
            if distribution[tag] >= 50:
                new.append(tag)
        train_data['tag'].iloc[idx_tag_1] = new

        idx_tag_1 += 1

    idx_tag_2 = 0
    # length = []
    for doc in train_data['tag']:
        if len(doc) > config.max_label_length:
            train_data['tag'].iloc[idx_tag_2] = doc[:3]
        # length.append(len(train_data['tag'].iloc[idx_tag_2]))
        # for tag in doc:
        #     if distribution[tag] < 50:
        #         idx_list.append(idx_tag)

        idx_tag_2 += 1
    # print(Counter(length))

    train_data = train_data.drop(idx_list)

    print(train_data.shape)


    return train_data


def data_analysis(train_data):
    f_tag = flatten(train_data['tag'])

    # number of classes
    tag_set = set(f_tag)
    print('number of classes:', len(tag_set))

    # data distribution
    distribution = Counter(f_tag)
    print('\n', distribution,'\n') # we can see the data is imbalanced
    print('average number of documents each class: ', np.mean(list(distribution.values())))

    # vocab
    train_data = train_data['all_text']
    vocab = Counter(flatten(train_data))
    print('size of vocab: ',len(vocab))

    # doc length
    length = []
    for doc in train_data:
        length.append(len(doc))
    plt.hist(length)
    # plt.show()
    len_distri = Counter(length)
    ave = sum(length) / len(length)
    ave = math.ceil(ave)

    count = 0
    for lens in length:
        if lens <= 300:
            count += 1

    print('percentage of doc whose length < 300:', count/len(length))
    print('average length of doc: ', ave)
    print('distribution of length:', len_distri)
    print('max and min length:', max(length), min(length))

    # frequency of words
    count = 0
    for k, v in vocab.items():
        if v > 30:
            count += 1
    print('percentage of word whose count >30:', count, count / len(vocab), '\n')


    return vocab, tag_set

# 
# def save_as_tsv(train_data):
#     train_data.to_csv("train_data.tsv", index=False, sep='\t')

if __name__ == '__main__':
    train_data= read_data()
    train_data = data_clean(train_data)
    vocab, tag_set = data_analysis(train_data)
    # save_as_tsv(train_data)
    # print(train_data.head(8))


# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from config import Config

config = Config()
kernel_sizes = [1,2,3,4]
kernel_sizes2 = [1,2,3,4]

# class TextCNN(nn.Module):
# #
# #     def __init__(self, config):
# #         super(TextCNN, self).__init__()
# #         self.config = config
# #         self.out_channel = config.out_channel
# #         self.conv3 = nn.Conv2d(1, 100, (3, config.word_embedding_dimension))
# #         self.conv4 = nn.Conv2d(1, 100, (4, config.word_embedding_dimension))
# #         self.conv5 = nn.Conv2d(1, 100, (5, config.word_embedding_dimension))
# #         self.Max3_pool = nn.MaxPool2d((self.config.sentence_max_size-3+1, 1))
# #         self.Max4_pool = nn.MaxPool2d((self.config.sentence_max_size-4+1, 1))
# #         self.Max5_pool = nn.MaxPool2d((self.config.sentence_max_size-5+1, 1))
# #         # self.dropout = nn.Dropout(dropout)
# #         self.linear1 = nn.Linear(3, config.label_num)
# #
# #     def forward(self, x):
# #         batch = x.shape[0]
# #         # Convolution
# #         x1 = F.relu(self.conv3(x))
# #         x2 = F.relu(self.conv4(x))
# #         x3 = F.relu(self.conv5(x))
# #
# #         # Pooling
# #         x1 = self.Max3_pool(x1)
# #         x2 = self.Max4_pool(x2)
# #         x3 = self.Max5_pool(x3)
# #
# #         # capture and concatenate the features
# #         x = torch.cat((x1, x2, x3), -1)
# #         x = x.view(batch, 1, -1)
# #
# #         # project the features to the labels
# #         x = self.linear1(x)
# #         x = x.view(-1, self.config.label_num)
# #
# #         return x
# #
# #
# # if __name__ == '__main__':
# #     print('running the TextCNN...')

# class TextCNN(nn.Module):
#
#     def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=100, kernel_sizes=(2, 3, 4), dropout=0.5):
#         super(TextCNN, self).__init__()
#
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])
#
#         # kernal_size = (K,D)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)
#
#     def init_weights(self, pretrained_word_vectors, is_static=False):
#         self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
#         if is_static:
#             self.embedding.weight.requires_grad = False
#
#     def forward(self, inputs, is_training=False):
#         inputs = self.embedding(inputs).unsqueeze(1)  # (B,1,T,D)
#         inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
#         inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
#
#         concated = torch.cat(inputs, 1)
#
#         if is_training:
#             concated = self.dropout(concated)  # (N,len(Ks)*Co)
#         out = self.fc(concated)
#         return F.log_softmax(out, 1)

class TextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=200, kernel_sizes=kernel_sizes, dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # B*D*V


        convs = [nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=kernel_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(kernel_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=kernel_dim,
                      out_channels=kernel_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(kernel_dim),
            nn.ReLU(inplace=True), # B*K*V
            nn.MaxPool1d(kernel_size=(config.max_sent_length - kernel_size*2 + 2))
        )
            for kernel_size in kernel_sizes]
        self.convs = nn.ModuleList(convs)

        # self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * (kernel_dim), config.hidden_linear_size),
            nn.BatchNorm1d(config.hidden_linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_linear_size, output_size)
        )

        # self.linear = nn.Linear(len(kernel_sizes) * (kernel_dim), config.hidden_linear_size)
        # self.batchn = nn.BatchNorm1d(config.hidden_linear_size)
        # self.relu = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(config.hidden_linear_size, output_size)
        # print(output_size)

        # kernal_size = (K,D)
        # self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False


    def forward(self, inputs, is_training=False):
        inputs = self.embedding(inputs)  # (B,1,T,D)
        # print('emb:', inputs.data.shape)
        inputs = [conv(inputs.permute(0, 2, 1)) for conv in self.convs]
        # print('conv:', [i.data.shape for i in inputs])

        # inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        # inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
        #
        concated = torch.cat(inputs, 1)
        # print('concat:', concated.data.shape)
        #
        # if is_training:
        #     concated = self.dropout(concated)  # (N,len(Ks)*Co)
        # out = self.fc(concated)
        reshaped = concated.view(concated.size(0), -1)
        # print('reshape:', reshaped.data.shape)
        logits = self.fc((reshaped))
        # o = self.linear(reshaped)
        # print('o:', o.data.shape)
        # o = self.batchn(o)
        # print('o:', o.data.shape)
        # o = self.relu(o)
        # print('o:', o.data.shape)
        # logits = self.linear2(o)
        # print('fc:', logits.data.shape)
        # return F.log_softmax(out, 1)
        return logits

class TextCNN_Adaptive(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=200, kernel_sizes=kernel_sizes, dropout=0.5):
        super(TextCNN_Adaptive, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # B*D*V


        convs = [nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=kernel_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(kernel_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=kernel_dim,
                      out_channels=kernel_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(kernel_dim),
            nn.ReLU(inplace=True), # B*K*V
            # nn.MaxPool1d(kernel_size=(config.max_sent_length - kernel_size*2 + 2))
            nn.AdaptiveMaxPool1d(config.max_k) # adaptive pooling  并不是严格的kmax，也不是分片max， 比较奇怪
        )
            for kernel_size in kernel_sizes]
        self.convs = nn.ModuleList(convs)

        # self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * (kernel_dim) * config.max_k, config.hidden_linear_size),
            nn.BatchNorm1d(config.hidden_linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_linear_size, output_size)
        )


        # kernal_size = (K,D)
        # self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False

    def forward(self, inputs, is_training=False):
        inputs = self.embedding(inputs)  # (B,1,T,D)
        # print('emb:', inputs.data.shape)
        inputs = [conv(inputs.permute(0, 2, 1)) for conv in self.convs]

        # print('conv:', [i.data.shape for i in inputs])
        # inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        # inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
        #
        concated = torch.cat(inputs, 1)
        # print('concat:', concated.data.shape)
        #
        # if is_training:
        #     concated = self.dropout(concated)  # (N,len(Ks)*Co)
        # out = self.fc(concated)
        reshaped = concated.view(concated.size(0), -1)
        # print('reshape:', reshaped.data.shape)
        logits = self.fc((reshaped))

        # print('fc:', logits.data.shape)
        # return F.log_softmax(out, 1)
        return logits


class TextCNN_Kmax(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=200, kernel_sizes=kernel_sizes, dropout=0.5):
        super(TextCNN_Kmax, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # B*D*V


        convs = [nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=kernel_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(kernel_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=kernel_dim,
                      out_channels=kernel_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(kernel_dim),
            nn.ReLU(inplace=True), # B*K*V
        )
            for kernel_size in kernel_sizes]
        self.convs = nn.ModuleList(convs)

        # self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * (kernel_dim) * config.max_k, config.hidden_linear_size),
            nn.BatchNorm1d(config.hidden_linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_linear_size, output_size)
        )

        # self.linear = nn.Linear(len(kernel_sizes) * (kernel_dim), config.hidden_linear_size)
        # self.batchn = nn.BatchNorm1d(config.hidden_linear_size)
        # self.relu = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(config.hidden_linear_size, output_size)
        # print(output_size)

        # kernal_size = (K,D)
        # self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]

        return x.gather(dim, index)

    def forward(self, inputs, is_training=False):
        inputs = self.embedding(inputs)  # (B,1,T,D)
        # print('emb:', inputs.data.shape)
        inputs = [conv(inputs.permute(0, 2, 1)) for conv in self.convs]
        inputs = [self.kmax_pooling(x, 2, config.max_k) for x in inputs]
        # print('conv:', [i.data.shape for i in inputs])
        # inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        # inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)
        #
        concated = torch.cat(inputs, 1)
        # print('concat:', concated.data.shape)
        #
        # if is_training:
        #     concated = self.dropout(concated)  # (N,len(Ks)*Co)

        # out = self.fc(concated)
        reshaped = concated.view(concated.size(0), -1)
        # print('reshape:', reshaped.data.shape)
        logits = self.fc((reshaped))

        # print('fc:', logits.data.shape)
        # return F.log_softmax(out, 1)
        return logits


class TextCNN_Attention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=200, kernel_sizes=kernel_sizes, dropout=0.5):
        super(TextCNN_Attention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # B*D*V


        convs = [nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=kernel_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(kernel_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=kernel_dim,
                      out_channels=kernel_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(kernel_dim),
            nn.ReLU(inplace=True), # B*K*V
            # nn.MaxPool1d(kernel_size=(config.max_sent_length - kernel_size*2 + 2))
            # nn.AdaptiveMaxPool1d(config.max_k) # 效果跟k_max似乎是一样的
            Attention_layer(config.max_sent_length - kernel_size * 2 + 2),
            nn.Linear(config.max_sent_length - kernel_size * 2 + 2, 200)
        )
            for kernel_size in kernel_sizes]
        self.convs = nn.ModuleList(convs)


        # self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])


        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * (kernel_dim), config.hidden_linear_size),
            nn.BatchNorm1d(config.hidden_linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_linear_size, output_size)
        )

        # kernal_size = (K,D)
        # self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)

    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False

    # def attention(self, inputs):


    def forward(self, inputs, is_training=False):
        inputs = self.embedding(inputs)  # (B,1,T,D)
        # print('emb:', inputs.data.shape)
        inputs = [conv(inputs.permute(0, 2, 1)) for conv in self.convs]
        # print('conv:', [i.data.shape for i in inputs])

        # inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        # inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # [(N,Co), ...]*len(Ks)

        concated = torch.cat(inputs, 1)
        # print('concat:', concated.data.shape)
        #
        # if is_training:
        #     concated = self.dropout(concated)  # (N,len(Ks)*Co)
        # out = self.fc(concated)
        reshaped = concated.view(concated.size(0), -1)
        # print('reshape:', reshaped.data.shape)
        logits = self.fc((reshaped))

        # print('fc:', logits.data.shape)
        # return F.log_softmax(out, 1)
        return logits


class Attention_layer(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.hidden_dim = input_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        # print('1', encoder_outputs.data.shape)
        energy = self.projection(encoder_outputs)
        # print('2', energy.data.shape)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # print('3', weights.data.shape, weights.unsqueeze(-1).data.shape)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = torch.bmm(encoder_outputs.transpose(1, 2), weights.unsqueeze(-1)).squeeze(-1)
        # print('4', outputs.data.shape)
        return outputs



if __name__ == '__main__':
    cnn = TextCNN(50000, 300, 100)
    cnn_adap = TextCNN_Adaptive(50000, 300, 100)
    cnn_kmax = TextCNN_Kmax(50000, 300, 100)
    cnn_atte = TextCNN_Attention(50000, 300, 100)
    x = torch.autograd.Variable(torch.arange(0, 15000).view(50, 300)).long()
    # o_1 = cnn(x)
    # o_2 = cnn_adap(x)
    # o_3 = cnn_kmax(x)
    o_4 = cnn_atte(x)
    print(cnn_atte)
    print(o_4.data.shape)
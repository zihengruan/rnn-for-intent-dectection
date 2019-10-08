import torch
import torch.nn as nn
import os
import pickle
from data import *
from test_data import *
from model import Encoder,Decoder
import numpy as np
import csv

USE_CUDA = torch.cuda.is_available()

class Check:
    def __init__(self):
        self.train_data = []
        self.word2index = {}
        self.tag2index = {}
        self.intent2index = {}
        self.test_data = []
        self.test_count = int()
        self.test_error_count = 0
        self.test_error = []
        self.test_error_rate = float()
        with open('./data/processed_train_data.pkl', 'rb') as f:
            self.train_data, self.word2index, self.tag2index, self.intent2index = pickle.load(f)  # , encoding='latin1'
        with open('./data/intent2index.csv','w') as ii:
            writer = csv.writer(ii)
            for key, value in self.intent2index.items():
                writer.writerow([key, value])
        # with open('./data/atis.test.w-intent.iob','r') as test_file:
        self.test_data = preprocessing_test_data('./data/atis.test.w-intent.iob', self.word2index, self.tag2index, self.intent2index,length=60)
        # print(self.test_data)

        self.intent2index_list = list(self.intent2index.keys())

        self.test_raw_input, self.test_truth = getting_raw_input('./data/atis.test.w-intent.iob', length=60)
        # print(self.test_truth)

    def test(self,encoder_in,decoder_in):
        # #初始化
        # encoder = Encoder(len(self.word2index), embedding_size = 64, hidden_size = 64)
        # decoder = Decoder(len(self.tag2index), len(self.intent2index), len(self.tag2index)// 3, hidden_size = 64*2)
        #
        # encoder.init_weights()
        # decoder.init_weights()
        #
        # #载入模型参数
        # encoder.load_state_dict(torch.load('models/jointnlu-encoder.pkl'))
        # decoder.load_state_dict(torch.load('models/jointnlu-decoder.pkl'))
        # encoder = torch.load('models/jointnlu-encoder.pkl')
        # decoder = torch.load('models/jointnlu-decoder.pkl')
        encoder = encoder_in
        decoder = decoder_in

        test_in = self.test_data
        # print(test_in)
        self.test_count = len(self.test_raw_input)
        # print(self.test_count)

        for i in range(self.test_count):
        # for i in range(100):
            x = torch.cat([test_in[i]])
            x_mask = torch.cat([Variable(
                torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(
                torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x]).view(1, -1)
            # print(i)
            # print(x)
            output, hidden_c = encoder(x, x_mask)
            start_decode = Variable(torch.LongTensor([[self.word2index['<SOS>']] * 1])).cuda().transpose(1,0) \
                if USE_CUDA else Variable(torch.LongTensor([[self.word2index['<SOS>']] * 1])).transpose(1, 0)

            tag_score, intent_score = decoder(start_decode, hidden_c, output, x_mask)

            intent_score_np = intent_score.data.numpy()
            _, intent_score_np_index = np.where(intent_score_np == np.max(intent_score_np))

            z = intent_score_np_index[0]
            # y = test_intent[i].data[0].item()
            try:
                y = self.intent2index[self.test_truth[i]]
            except KeyError as ke:
                y = -1
                print("i = " + str(i))
                print("OS error: {0}".format(ke))
            if z != y:
                self.test_error_count += 1
                _ = ' '.join(self.test_raw_input[i])
                self.test_error.append(zip([_], [self.test_truth[i]], [self.intent2index_list[z]]))#input,truth,predition

        self.test_error_rate = self.test_error_count / self.test_count
        with open('./data/error_result.csv','w') as error_result:
            writer = csv.writer(error_result)
            for item in self.test_error:
                writer.writerow(zip(*item))
            writer.writerow([self.test_error_count])
            writer.writerow([self.test_error_rate])

        print(self.test_error_count)
        print(self.test_error_rate)


if __name__ == '__main__':
    t = Check()
    t.test()
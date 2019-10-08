import torch
from torch.autograd import Variable
from collections import Counter
import pickle
import random
import os

USE_CUDA = torch.cuda.is_available()


def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs)).cuda() if USE_CUDA else Variable(torch.LongTensor(idxs))
    return tensor


flatten = lambda l: [item for sublist in l for item in sublist]


def getting_raw_input(file_path, length):
    """
    atis.test.w-intent.iob
    """

    try:
        train = open(file_path, "r").readlines()
        # print("Successfully load data. # of set : %d " % len(train))
    except:
        print("No such file!!!!!!!!!!!")
        return None, None, None, None

    try:
        train = [t[:-1] for t in train]
        train = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t
                 in train]
        train = [[t[0][1:-1], t[1][1:], t[2]] for t in train]

        seq_in, seq_out, intent = list(zip(*train))
        # vocab = set(flatten(seq_in))
        # slot_tag = set(flatten(seq_out))
        # intent_tag = set(intent)
    except:
        print(
            "Please, check data format! It should be 'raw sentence \t BIO tag sequence intent'. The following is a sample.")
        print(
            "BOS i want to fly from baltimore to dallas round trip EOS\tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight")
        return None, None, None, None

    return seq_in, intent

def preprocessing_test_data(file_path, word2index_in, tag2index_in, intent2index_in, length):
    try:
        train = open(file_path,"r").readlines()
        # print("Successfully load data. # of set : %d " % len(train))
    except:
        print("No such file!")
        return None,None,None,None

    try:
        train = [t[:-1] for t in train]
        train = [[t.split("\t")[0].split(" "),t.split("\t")[1].split(" ")[:-1],t.split("\t")[1].split(" ")[-1]] for t in train]
        train = [[t[0][1:-1],t[1][1:],t[2]] for t in train]

        seq_in,seq_out, intent = list(zip(*train))

        #TODO KILL DIGIT


        vocab = set(flatten(seq_in))
        slot_tag = set(flatten(seq_out))
        intent_tag = set(intent)
        # print("# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}".format(vocab=len(vocab),slot_tag=len(slot_tag),intent_tag=len(intent_tag)))
    except:
        print("Please, check data format! It should be 'raw sentence \t BIO tag sequence intent'. The following is a sample.")
        print("BOS i want to fly from baltimore to dallas round trip EOS\tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight")
        return None,None,None,None

    sin = []
    sout = []

    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sin.append(temp)

        temp = seq_out[i]
        if len(temp) < length:
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sout.append(temp)

    word2index = word2index_in
    # for token in vocab:
    #     if token not in word2index.keys():
    #         word2index[token]=len(word2index)

    tag2index = tag2index_in
    # for tag in slot_tag:
    #     if tag not in tag2index.keys():
    #         tag2index[tag] = len(tag2index)

    intent2index = intent2index_in
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)

    train = list(zip(sin, sout, intent))

    train_data=[]

    for tr in train:

        temp = prepare_sequence(tr[0],word2index)
        temp = temp.view(1,-1)

        # temp2 = prepare_sequence(tr[1],tag2index)
        # temp2 = temp2.view(1,-1)

        # temp3 = Variable(torch.LongTensor([intent2index[tr[2]]])).cuda() if USE_CUDA else Variable(torch.LongTensor([intent2index[tr[2]]]))

        # train_data.append((temp,temp2,temp3))
        train_data.append(temp)

    return train_data
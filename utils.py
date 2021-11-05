#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np


def trans_item(item, itm_dict, mode='word2idx'):
    try:
        return itm_dict[item]
    except:
        if mode == 'word2idx':
            return 0
        else:
            # mode == 'idx2word'
            return '<unknow>'


def trans_sent(sent, itm_dict, mode='word2idx'):
    outs = []
    for word in sent:
        outs.append(trans_item(word, itm_dict, mode=mode))
    return outs


def trans_batch_sents(batch_sents, itm_dict, mode='word2idx'):
    outs = []
    for sent in batch_sents:
        outs.append(trans_sent(sent, itm_dict, mode))
    return outs


def trans_data(data, vocab_dict, mode='word2idx'):
    out_data = []
    for inp, target in data:
        trans_inp = trans_sent(inp, vocab_dict, mode)
        trans_target = trans_sent(target, vocab_dict, mode)
        out_data.append((trans_inp, trans_target))

    return out_data


word_vocab = ['Br', 'Cl', 'Si', '@@', 'nH', 'Pt', 'Pd', \
              'Li', 'Be', 'Ne', 'He', 'Se', 'se', \
              'Na', 'Mg', 'Al', 'Ar', 'Ca', 'Fe', 'Cu', 'Ge', 'Kr',\
              'Sc', 'Ti', 'Cr', 'Mn', 'Ni', 'Zn', 'Ga', 'As', 'Se', 'Sn']

def maxMatch(word_vocab, sent, maxL=2):
    endPos = len(sent)
    startPos = 0
    fenci = []
    while startPos < endPos:
        L = maxL
        while L > 0:
            subsent = sent[startPos: startPos+L]
            if subsent in word_vocab or L == 1:
                fenci.append(subsent)
                startPos += L
                break
            L -= 1
    return fenci


def read_data(data_dir, 
              sign_start='START!', sign_end='END!', 
              _all_prefix=['train', 'dev', 'test']):
    data = {}
    all_words = []
    for _pre in _all_prefix:
        # read source
        with open(data_dir+'/'+_pre+'.source', 'r') as ifile:
            src_sents = []
            for line in ifile:
                smiles = line.strip().split('>')[-1]
                fenci = maxMatch(word_vocab, smiles)
                fenci = [sign_start] + fenci + [sign_end]
                src_sents.append(fenci)
                # for vacabulary set
                if _pre == 'train':
                    all_words += fenci
        
        # read target
        with open(data_dir+'/'+_pre+'.target', 'r') as ifile:
            tgt_sents = []
            for line in ifile:
                smiles = line.strip()
                fenci = maxMatch(word_vocab, smiles)
                fenci = [sign_start] + fenci + [sign_end]
                tgt_sents.append(fenci)
                # for label set
                if _pre == 'train':
                    all_words += fenci

        _pre_data = []
        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            _pre_data.append((src_sent, tgt_sent))

        data[_pre] = _pre_data

    return data, set(all_words), sign_start, sign_end


def inverse_dict(in_dict):
    return dict((v, k) for k, v in in_dict.items())


def get_batch_data_approx(data, pos, approx_num):
    batch_inputs = []
    batch_targets = []
    maxT = -1
    
    new_batch_size = 0
    while True:
        try:
            in_sent, out_sent = data[pos+new_batch_size]
        except:
            return batch_inputs, batch_targets, new_batch_size, maxT * new_batch_size

        ###
        new_batch_size += 1
        batch_inputs.append(in_sent)
        batch_targets.append(out_sent)
        
        ## continue?
        maxCurr = max(len(in_sent), len(out_sent))
        new_maxT = max(maxCurr, maxT)
        numTokens = new_maxT * new_batch_size
        if abs(numTokens - approx_num) <= 30:
            break
        elif numTokens > approx_num:
            # remove the last one
            batch_inputs.pop()
            batch_targets.pop()
            new_batch_size -= 1
            break

        ## update maxT
        maxT = new_maxT
    
    return batch_inputs, batch_targets, new_batch_size, maxT * new_batch_size

def batch_pad(data, val=0, front=False):
    # get maximun length of data
    T = max(map(len, data))

    padded_data = []
    for sent in data:
        L = len(sent)
        _num = T - L
        # front padding
        if front:
            padded_data.append(
                np.pad(sent, (_num, 0), 'constant', constant_values=(val))
            )
        else:
            padded_data.append(
                np.pad(sent, (0, _num), 'constant', constant_values=(val))
            )
    return np.array(padded_data)


def over_print(content):
    """
    overwrite print
    
    Arguments:
        content {[type]} -- [description]
    """

    sys.stdout.write('\r'+content)
    sys.stdout.flush()


def get_batch_data(data, pos, batch_size, word_size=None):
    """
    get batch size data
    
    Arguments:
        data {[type]} -- [description]
        pos {[type]} -- [description]
        batch_size {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    batch_inputs = []
    batch_targets = []
    batch_in_pos = []
    batch_out_pos = []
    for n in range(batch_size):
        try:
            in_sent, out_sent = data[pos+n]
        except:
            return batch_inputs, batch_in_pos, batch_targets, batch_out_pos, n
        
        # for outputs
        if word_size:
            out_sent.append(word_size+n)

        batch_inputs.append(in_sent)
        batch_targets.append(out_sent)
        batch_in_pos.append(np.arange(1, len(in_sent)+1))
        batch_out_pos.append(np.arange(1, len(out_sent)+1))
    return batch_inputs, batch_in_pos, batch_targets, batch_out_pos, batch_size

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import time
import json
import argparse
import tensorflow as tf
import numpy as np
#from sklearn.model_selection import train_test_split
from transformer import Transformer
from utils import *


parser = argparse.ArgumentParser(description='training Transformer')
parser.add_argument('--data_dir', action='store', type=str, 
                    default='data/BV/', help='data directory.')
parser.add_argument('--max_epochs', '-max_ep', action='store', type=int,
                    default=200000, help='maximum epochs for training.')
parser.add_argument('--model_dir', action='store', type=str,
                    default='model/', help='directory to save model')
parser.add_argument('--model_name', action='store', type=str,
                    default='ChemTrm', help='name of model')
parser.add_argument('--approx_num', action='store', type=int, default=3144,
                    help='batch size for training.')


def eval(test_model, 
         sess, 
         writer, tf_acc, acc_summary, global_step,
         val_data, max_length,
         approx_num, 
         code_of_start, 
         code_of_end, 
         code_of_pad):
    np.random.shuffle(val_data)
    N = len(val_data)
    pos = 0
    correct = 0
    approx_num *= 3

    def post_proc(sent_ids):
        out = []
        for idx in sent_ids:
            if idx == code_of_start:
                continue
            elif idx == code_of_end:
                break
            else:
                out.append(idx)
        return out

    def is_same(sent_a, sent_b):
        if len(sent_a) != len(sent_b):
            return False

        for a, b in zip(sent_a, sent_b):
            if a != b:
                return False
        return True

    while True:
        # get batch data
        batch_inputs, batch_targets, new_batch_size, _ = \
                                get_batch_data_approx(val_data, pos, approx_num)
        
        # padd and get position
        # front padding for inputs of encoder
        batch_inputs = batch_pad(batch_inputs, val=code_of_pad, front=False)

        pred_sents = test_model.predict(sess, batch_inputs, max_length)

        ##### calc
        for pred, tgt in zip(pred_sents, batch_targets):
            pred = post_proc(pred)
            tgt = post_proc(tgt)
            correct += 1 if is_same(pred, tgt) else 0

        print('{}/{}'.format(pos+new_batch_size, N))
        if (pos+new_batch_size) >= N:
            break

        # update pos
        pos += new_batch_size

    acc = correct / float(N)
    acc_summary_str, gs = sess.run([acc_summary, global_step], {tf_acc: acc})
    writer.add_summary(acc_summary_str, global_step=gs)
    print('eval process done!')


if __name__ == '__main__':
    args = parser.parse_args()

    # get arguments
    data_dir = args.data_dir
    max_epochs = args.max_epochs
    model_dir = args.model_dir
    model_name = args.model_name
    approx_num = args.approx_num

    # read data from files
    data, vocab, sign_start, sign_end = read_data(data_dir)
    train_data = data['train']
    val_data = data['dev']
    test_data = data['test']
    all_data = train_data + val_data + test_data
    max_sent_len_in = max(map(len, [sent[0] for sent in all_data]))
    max_sent_len_out = max(map(len, [sent[1] for sent in all_data]))
    # print(max_sent_len_in, max_sent_len_out)
    max_sent_len = max(max_sent_len_in, max_sent_len_out)
    print('the largest sentence size={}'.format(max_sent_len))

    # reform training data and valdata
    #train_data = train_data + val_data
    #train_data, val_data = train_test_split(train_data, test_size=.0)

    print('training data size={}'.format(len(train_data)))
    print('validation data size={}'.format(len(val_data)))
    print('test data size={}'.format(len(test_data)))

    # add sign of pad for label set
    sign_pad = '<pad>'
    # labelset = labelset | set([sign_pad])

    ### load word2idx
    try:
        with open(model_dir+'word2idx.json', 'r') as ifile:
            json_str = ifile.readline()

        vocab_set = json.loads(json_str)
        print('vocab loaded.')
    except:
        print('vocab not found!!!!')
        exit(-1)

    idx2word_vocab = inverse_dict(vocab_set)
    vocab_size = len(vocab_set)
    print(vocab_set)
    print(idx2word_vocab)
    print('vocabulary size={}'.format(vocab_size))
    # word2idx
    train_data = trans_data(train_data, vocab_set)
    val_data = trans_data(val_data, vocab_set)
    test_data = trans_data(test_data, vocab_set)

    # setup parameters
    n_heads = 8
    emb_dim = 256
    num_layers = 6
    FFN_inner_units = 2048
    dropout_keep_prob = 0.7

    global_step = tf.get_variable('global_step', initializer=tf.constant(0.))
    
    kwargs = {
        'voca_size': vocab_size,
        'label_size': vocab_size,
        'code_of_start': vocab_set[sign_start],
        'code_of_end': vocab_set[sign_end],
        'code_of_pad': vocab_set[sign_pad],
        'num_layers_enc': num_layers,
        'num_layers_dec': num_layers,
        'emb_dim': emb_dim,
        'n_heads': n_heads,
        'dropout_keep_prob': dropout_keep_prob,
        'FFN_inner_units': FFN_inner_units
    }

    # build graph
    def build_model_fn_wrapper(reuse=tf.AUTO_REUSE, is_train=True):
        model = Transformer(is_train=is_train, reuse=reuse, **kwargs)
        model._build_graph()
        if model.is_train:
            model._build_loss()
            model._build_train_op(global_step=global_step)
        return model
    
    model = build_model_fn_wrapper(is_train=True)
    # model for test
    test_model = build_model_fn_wrapper(is_train=False)

    #### build graph for acc
    tf_acc = tf.placeholder(tf.float32)
    acc_summary = tf.summary.scalar('eval_acc', tf_acc)
    
    # create session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.InteractiveSession(config=sess_config)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # set a summary writer
    writer = tf.summary.FileWriter(model_dir, sess.graph)

    # read checkpoint
    ckpt = tf.train.latest_checkpoint(checkpoint_dir=model_dir)
    if ckpt is None:
        sess.run(tf.global_variables_initializer())
        print('model initialized.')
    else:
        saver.restore(sess, ckpt)
        print('latest model -- {} has been loaded.'.format(ckpt))

    # training procedure
    N = len(train_data)
    beam_size = 4
    log_n_steps = 10
    _last_t = -1.
    for epoch in range(max_epochs):
        # shuffle training data
        np.random.shuffle(train_data)
        losses = []
        pos = 0
        log_rec = 0

        while True:
            # get batch data
            batch_inputs, batch_targets, new_batch_size, numTokens = \
                                  get_batch_data_approx(train_data, pos, approx_num)
            
            # padd and get position
            # front padding for inputs of encoder
            batch_inputs = batch_pad(batch_inputs, val=vocab_set[sign_pad], front=False)
            # backward padding for targets (inputs of decoder)
            batch_targets = batch_pad(
                batch_targets,
                val=vocab_set[sign_pad],
                front=False
            )
            
            loss, train_summary = model.update(sess, batch_inputs, batch_targets)
            losses.append(loss)

            # terminate condition
            # if new_batch_size < batch_size:
            #     break
            if (pos+new_batch_size) >= N:
                break

            # print information
            over_print('{}/{}'.format(pos+new_batch_size, N))
            # if (pos+new_batch_size) % (batch_size*10) == 0:
            if log_rec % log_n_steps == 0:
                over_print(
                    '{}/{}, mean_loss={:.5f}'.format(
                        pos+new_batch_size, N, np.mean(losses)
                    )
                )
                writer.add_summary(train_summary, global_step=model.get_global_step(sess))
                print()
                losses = []
                np_lrs = []
                log_rec = 0
            
            # update pos
            pos += new_batch_size
            log_rec += 1
        
        # print last information
        over_print(
            '{}/{}, mean_loss={}'.format(pos+new_batch_size, N, np.mean(losses))
        )
        print()

        ### save model
        saver.save(sess, model_dir+model_name, global_step=global_step)
        print('epoch {} finished and model saved!'.format(epoch))

        _curr_t = time.time()
        if (_curr_t - _last_t) > 600. or _last_t == -1:
            #### validation
            eval(
                test_model, 
                sess,
                writer, tf_acc, acc_summary, global_step,
                val_data, max_sent_len, approx_num, 
                vocab_set[sign_start], vocab_set[sign_end], vocab_set[sign_pad]
            )
            _last_t = time.time() ## without eval time

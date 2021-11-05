
import json
import time
import tensorflow as tf
from transformer import Transformer
from utils import *

def get_smiles(sent_list, vocab_set, sign_start, sign_end, idx2word_vocab):
    smiles = ''
    for char in sent_list:
        if char == idx2word_vocab[vocab_set[sign_end]]:
            break
        elif char != idx2word_vocab[vocab_set[sign_start]]:
            smiles += char
        else:
            pass
    return smiles

if __name__ == '__main__':
    # get arguments
    data_dir = 'data/BV/'
    out_dir = 'out/'
    batch_size = 32
    model_dir = 'model/'
    model_name = 'ChemTrm'

    # read data from files
    sign_pad = '<pad>'
    data, vocab, sign_start, sign_end = read_data(data_dir)
    test_data = data['test']

    ### load word2idx
    with open(model_dir+'word2idx.json', 'r') as ifile:
        json_str = ifile.readline()

    vocab_set = json.loads(json_str)
    print('vocab loaded.')

    idx2word_vocab = inverse_dict(vocab_set)
    vocab_size = len(vocab_set)
    print('vocabulary size={}'.format(vocab_size))
    test_data = trans_data(test_data, vocab_set)

    # setup parameters
    n_heads = 8
    emb_dim = 256
    num_layers = 6
    FFN_inner_units = 2048
    dropout_keep_prob = 0.9

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

    # model for test
    test_model = Transformer(is_train=False, reuse=False, **kwargs)
    test_model._build_graph()

    # create session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.InteractiveSession(config=sess_config)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # read checkpoint
    ckpt = tf.train.latest_checkpoint(checkpoint_dir=model_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        print('latest model -- {} has been loaded.'.format(ckpt))
    else:
        print('no model found!')
        exit(-1)

    writeToFile = True
    if writeToFile:
        greedy_out_filename = out_dir + 'greedy.out'
        greedy_out_file = open(greedy_out_filename, 'w')

    pos = 0
    N = len(test_data)
    beam_size = 10
    _s_t = time.time()
    while True:
        # get batch data
        batch_inputs, _, batch_targets, _, new_batch_size = \
                            get_batch_data(test_data, pos, batch_size)
        
        # padd and get position
        # front padding for inputs of encoder
        batch_inputs = batch_pad(batch_inputs, val=vocab_set[sign_pad], front=False)
    
        #### greedy decode
        batch_greedy_pred = test_model.predict(sess, batch_inputs, max_length=200)
        print('{}/{}'.format(pos + new_batch_size, N))

        for b in range(new_batch_size):
            greedy_sent_list = trans_sent(batch_greedy_pred[b, :], idx2word_vocab, mode='idx2word')
            #print(greedy_sent_list)
            greedy_out_file.write(get_smiles(greedy_sent_list, vocab_set, sign_start, sign_end, idx2word_vocab) + '\n')

        # terminate condition
        if new_batch_size < batch_size or (pos+new_batch_size==N):
            break

        # update pos
        pos += new_batch_size

    _e_t = time.time()
    print('all test done, used {:.2f} secs.'.format(_e_t-_s_t))

    if writeToFile:
        greedy_out_file.close()

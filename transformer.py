#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
from transformer_modules import * 


def embedding(word_inds, 
              vocab_size, 
              emb_dim, 
              initializer=None,
              scope='embedding',
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        emb_table = tf.get_variable(
            name=scope+'_table',
            shape=[vocab_size, emb_dim],
            initializer=initializer,
            dtype=tf.float32
        )

        embs = tf.nn.embedding_lookup(emb_table, word_inds)
        
        return embs, emb_table


def _encoder(embs,
             attn_bias,
             num_layers=6,
             n_heads=8,
             dropout_keep_prob=.9,
             is_training=True,
             n_inner=2048,
             summaries=None,
             attn_tensors=None,
             scope='encoder'):
    """
    Implementation of encoder for transformer
    """

    with tf.variable_scope(scope):
        hidd = add_timing_signal_1d(embs)

        if is_training:
            hidd = tf.nn.dropout(hidd, keep_prob=dropout_keep_prob)
        
        # multiple layers
        for n in range(num_layers):
            with tf.variable_scope('layer_{}'.format(n)):
                # first sublayer
                hidd = multihead_attention_module(
                    hidd, ctx=None, 
                    n_heads=n_heads,
                    use_bias=False,
                    keep_prob=dropout_keep_prob,
                    is_training=is_training,
                    attn_bias=attn_bias,
                    summaries=summaries,
                    attn_tensors=attn_tensors
                )

                # second sublayer
                hidd = pos_wise_FFN_module(
                    hidd,
                    n_inner=n_inner,
                    keep_prob=dropout_keep_prob,
                    is_training=is_training
                )

        return tf.contrib.layers.layer_norm(hidd, begin_norm_axis=-1)


def _decoder(num_outputs,
             ctx,  # context form encoder,
             embs,
             self_attn_bias,
             enc_dec_attn_bias=None,
             num_layers=6,
             n_heads=8,
             dropout_keep_prob=.9,
             is_training=True,
             n_inner=2048,
             final_proj_weights=None,
             summaries=None,
             attn_tensors=None,
             caches=None,
             scope='decoder'):
    """
    Implementation of decoder
    """

    with tf.variable_scope(scope):
        
        if is_training:
            hidd = add_timing_signal_1d(embs)
            hidd = tf.nn.dropout(hidd, keep_prob=dropout_keep_prob)
        else:
            hidd = add_timing_signal_1d(embs, start_index=tf.shape(caches[0])[1]-1)

        # multiple layers
        for n in range(num_layers):
            with tf.variable_scope('layer_{}'.format(n)):
                # first sublayer
                hidd = multihead_attention_module(
                    hidd, ctx=None,
                    n_heads=n_heads,
                    use_bias=False,
                    keep_prob=dropout_keep_prob,
                    is_training=is_training,
                    attn_bias=self_attn_bias,
                    summaries=summaries,
                    attn_tensors=attn_tensors,
                    without_cache=False,
                    caches=caches,
                    cache_id=n,
                    scope='self_attention'
                )
                if not is_training:
                    hidd = tf.expand_dims(hidd[:, -1, :], axis=1)

                # second sublayer
                hidd = multihead_attention_module(
                    hidd, ctx=ctx,
                    n_heads=n_heads,
                    use_bias=False,
                    keep_prob=dropout_keep_prob,
                    is_training=is_training,
                    attn_bias=enc_dec_attn_bias,
                    summaries=summaries,
                    attn_tensors=attn_tensors,
                    scope='cross_attention'
                )

                # third sublayer
                hidd = pos_wise_FFN_module(
                    hidd,
                    n_inner=n_inner,
                    keep_prob=dropout_keep_prob,
                    is_training=is_training
                )

        ### post process
        hidd = tf.contrib.layers.layer_norm(hidd, begin_norm_axis=-1)

        if final_proj_weights is None:
            with tf.variable_scope('final_projection'):
                unnorm_out = tf.layers.dense(
                    inputs=hidd,
                    units=num_outputs,
                    activation=None
                )
        else:
            ### (B, T, E) x (V, E)
            unnorm_out = tf.tensordot(
                hidd, final_proj_weights, 
                axes=[[2], [1]], 
                name='final_projection'
            )

        return unnorm_out


class Transformer:
    def __init__(self, 
                 voca_size, 
                 label_size,
                 code_of_start,
                 code_of_end,
                 code_of_pad,
                 num_layers_enc=6,
                 num_layers_dec=6,
                 emb_dim=256,
                 n_heads=8,
                 dropout_keep_prob=.9,
                 FFN_inner_units=2048,
                 shared_embedding_and_softmax_weights=True,
                 is_train=True,
                 reuse=False,
                 summaries=[],
                 scope='transformer'):
        # get informations
        self.scope = scope
        self.reuse = reuse
        self.is_train = is_train
        self.voca_size = voca_size
        self.label_size = label_size
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.dropout_keep_prob = dropout_keep_prob
        self.FFN_inner_units = FFN_inner_units
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.code_of_start = code_of_start
        self.code_of_end = code_of_end
        self.code_of_pad = code_of_pad
        self.shared_embedding_and_softmax_weights = shared_embedding_and_softmax_weights
        self.summaries = summaries
        self.attn_tensors = []

    def forward(self, inputs, outputs):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            
            ### encoder self attention bias
            with tf.name_scope('enc_self_attn_bias'):
                ## (B, T) --> (B, 1, 1, T)
                enc_self_attn_bias = tf.to_float(tf.equal(inputs, self.code_of_pad))
                self.enc_self_attn_bias = tf.expand_dims(
                    tf.expand_dims(enc_self_attn_bias * -1e9, axis=1), axis=1
                )

            ### t2t decoder self attention bias lower triangle
            with tf.name_scope('dec_self_attn_bias_lower_triangle'):
                if self.is_train:
                    T = tf.shape(outputs)[1]
                else:
                    T = tf.shape(self.decoder_caches[0])[1]
                # band: (T, T) --> (1, 1, T, T)
                band = tf.matrix_band_part(tf.ones([T, T]), -1, 0)  ## lower triangular part
                band = tf.expand_dims(tf.expand_dims(band, axis=0), axis=0)
                dec_self_attn_bias = -1e9 * (1.0 - band)

            ### embeddings
            enc_embs, enc_lookup_table = embedding(
                inputs, self.voca_size, self.emb_dim,
                initializer=tf.random_normal_initializer(0.0, self.emb_dim**-0.5)
            )

            if self.shared_embedding_and_softmax_weights:
                with tf.name_scope('shared_target_embedding'):
                    dec_embs = tf.nn.embedding_lookup(enc_lookup_table, outputs)
                    ### scale
                    dec_embs *= self.emb_dim ** 0.5
                    
                final_proj_weights = enc_lookup_table
                self.dec_lookup_table = enc_lookup_table
            else:
                dec_embs, dec_lookup_table = embedding(
                    outputs, self.voca_size, self.emb_dim,
                    initializer=tf.random_normal_initializer(0.0, self.emb_dim**-0.5),
                    scope='target_embedding'
                )
                final_proj_weights = None
                self.dec_lookup_table = dec_lookup_table

            # encoding
            self.ctx = _encoder(
                embs=enc_embs,
                attn_bias=self.enc_self_attn_bias,
                num_layers=self.num_layers_enc,
                n_heads=self.n_heads,
                dropout_keep_prob=self.dropout_keep_prob,
                is_training=self.is_train,
                n_inner=self.FFN_inner_units,
                summaries=self.summaries,
                attn_tensors=self.attn_tensors
            )

            if self.is_train:
                ctx = self.ctx
                enc_dec_attn_bias = self.enc_self_attn_bias
                caches = None
            else:
                ctx = self.ctx_in
                enc_dec_attn_bias = self.enc_dec_attn_bias_in
                caches = self.decoder_caches

            # decoding, logits: (B, T, O)
            logits = _decoder(
                num_outputs=self.label_size,
                ctx=ctx,
                embs=dec_embs,
                self_attn_bias=dec_self_attn_bias,
                enc_dec_attn_bias=enc_dec_attn_bias,
                num_layers=self.num_layers_dec,
                n_heads=self.n_heads,
                dropout_keep_prob=self.dropout_keep_prob,
                is_training=self.is_train,
                n_inner=self.FFN_inner_units,
                final_proj_weights=final_proj_weights,
                summaries=self.summaries,
                attn_tensors=self.attn_tensors,
                caches=caches
            )

            if not self.is_train:
                self.next_caches = self.decoder_caches[-self.num_layers_dec:]
                self.decoder_caches = self.decoder_caches[:self.num_layers_dec]

            return logits

    def _build_graph(self):
        # build inputs
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        if self.is_train:
            self.outputs = tf.placeholder(tf.int32, [None, None], name='outputs')
        else:
            self.outputs = tf.placeholder(tf.int32, [None, 1], name='outputs')
            self.ctx_in = tf.placeholder(tf.float32, [None, None, self.emb_dim], name='ctx_in')
            self.enc_dec_attn_bias_in = tf.placeholder(tf.float32, [None, 1, 1, None], name='enc_dec_attn_bias_in')
            self.decoder_caches = []
            for i in range(self.num_layers_dec):
                self.decoder_caches.append(
                    tf.placeholder(tf.float32, [None, None, self.emb_dim], name='layer_cache_{}'.format(i))
                )

        self.logits = self.forward(self.inputs, self.outputs)

        if not self.is_train:
            # (B, T, O) --> (B, O) --> (B,)
            self.sample_id = tf.argmax(
                tf.nn.softmax(self.logits[:, -1, :], axis=-1),
                axis=-1
            )
            # (B, T, O) --> (B, O)
            self.log_soft = tf.nn.log_softmax(self.logits[:, -1, :], axis=-1)

    def _build_loss(self):
        # shift ouputs for decoder and targets
        batch_size = tf.shape(self.outputs)[0]
        targets = tf.concat(
            [
                self.outputs[:, 1:],
                tf.ones([batch_size, 1], dtype=tf.int32) * self.code_of_pad
            ],
            axis=-1
        )
        
        # setup loss
        istarget = tf.to_float(tf.not_equal(targets, self.code_of_pad))

        seq_loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.logits,
            targets=targets,
            weights=istarget,
            name='seq_loss'
        )

        self.loss = seq_loss
        self.summaries.append(tf.summary.scalar('loss', self.loss))

    def _build_train_op(self, opt=None, lr=1e-3, global_step=None):                
        # setup optimizer
        if opt is None:
            opt = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=0.9,
                beta2=0.98,
                epsilon=1e-9
            )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.minimize(
                loss=self.loss,
                global_step=global_step
            )
        self.global_step = global_step

        self.summary_op = tf.summary.merge(self.summaries)

    def update(self, sess, 
                inputs, 
                targets):
        """ update parameters of neural network """
        loss, _, summary_str = sess.run(
            [self.loss, self.train_op, self.summary_op],
            {
                self.inputs: inputs, self.outputs: targets
            }
        )
        return loss, summary_str

    def predict(self, sess, inputs, max_length):
        # initialize
        B = len(inputs)

        ### get ctx for all decode process
        ctx, enc_dec_attn_bias = sess.run([self.ctx, self.enc_self_attn_bias], {self.inputs: inputs})
        feed_dict = {self.ctx_in: ctx, self.enc_dec_attn_bias_in: enc_dec_attn_bias}

        ### fast greedy decoding
        end_array = np.ones([B]) * self.code_of_end
        outputs = np.ones([B, 1]) * self.code_of_start
        out = np.ones([B, 1]) * self.code_of_start

        run_list = [self.sample_id] + self.next_caches
        for t in range(max_length):
            # (B,)
            if t == 0:
                for idx in range(self.num_layers_dec):
                    feed_dict.update({
                        self.decoder_caches[idx]: np.zeros([B, 1, self.emb_dim])
                    })
            else:
                for idx in range(self.num_layers_dec):
                    feed_dict.update({
                        self.decoder_caches[idx]: caches[idx]
                    })
                # print(caches[0].shape)
            feed_dict.update({self.outputs: out.reshape([B, 1])})

            out, *caches = sess.run(run_list, feed_dict)
            for b in range(B):
                # last symbol is end
                if outputs[b, -1] == self.code_of_end:
                    # modify symbol of current
                    out[b] = self.code_of_end
            outputs = np.concatenate((outputs, out.reshape([B, 1])), axis=-1)
            if (out == end_array).all():
                break

        return outputs.astype(np.int32)

    def get_global_step(self, sess):
        """ get global step """
        return sess.run(self.global_step)

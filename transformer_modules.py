#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
from attention import multihead_attention


def multihead_attention_module(hidd, ctx=None,
                               num_units=None,
                               n_heads=8,
                               use_bias=False,
                               keep_prob=.9,
                               attn_keep_prob=.9,
                               is_training=False,
                               attn_bias=None,
                               summaries=None,
                               attn_tensors=None,
                               without_cache=True,
                               caches=[],
                               cache_id=None,
                               scope='multihead_attention_module'):
    with tf.variable_scope(scope):

        Q = tf.contrib.layers.layer_norm(hidd, begin_norm_axis=-1) 
        
        if ctx is None:
            if is_training or without_cache:
                K = Q
                V = Q
            else:
                ### for fast decode, start from second time
                assert caches != []
                # (B, t+1, E) & (B, 1, E)
                next_cache = tf.concat([caches[cache_id], Q], axis=1)
                # append to caches
                caches.append(next_cache)
                Q = next_cache[:, 1:, :]
                K = Q
                V = Q
        else:
            K = ctx
            V = ctx

        multihead_attn = multihead_attention(Q, K, V,
                                             num_units=num_units,
                                             n_heads=n_heads,
                                             use_bias=use_bias,
                                             keep_prob=attn_keep_prob,
                                             is_training=is_training,
                                             attn_bias=attn_bias,
                                             summaries=summaries,
                                             attn_tensors=attn_tensors)
        if is_training:
            multihead_attn = tf.nn.dropout(multihead_attn, keep_prob=keep_prob)

        # residual and layer normalization
        return multihead_attn + hidd


def _pos_wise_FFN(x, n_hidds, 
                  n_inner=2048,
                  keep_prob=0.9,
                  is_training=False,
                  scope='PW_FFN'):
    with tf.variable_scope(scope):
        fc1 = tf.layers.dense(
            inputs=x,
            units=n_inner,
            activation=tf.nn.relu
        )

        if is_training:
            fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

        out = tf.layers.dense(
            inputs=fc1,
            units=n_hidds,
            activation=None,
        )

        return out


def pos_wise_FFN_module(x, 
                        n_inner=2048,
                        keep_prob=0.9,
                        is_training=False,
                        scope='PW_FNN_module'):
    """ postion wise Feed forward networks sublayer module """
    with tf.variable_scope(scope):
        n_hidds = x.get_shape().as_list()[-1]

        norm_x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)

        pos_ffn = _pos_wise_FFN(norm_x, n_hidds, 
            n_inner=n_inner, keep_prob=keep_prob, is_training=is_training)

        if is_training:
            pos_ffn = tf.nn.dropout(pos_ffn, keep_prob=keep_prob)

        # residual and layer normalization
        return pos_ffn + x


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):

    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
        np.log(float(max_timescale) / float(min_timescale)) /
        tf.maximum(tf.to_float(num_timescales) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])

    ## signal: (1, T, C)
    return signal


def add_timing_signal_1d(x,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0,
                         name='add_timing'):
    with tf.name_scope(name):
        ## x: (B, T, E)
        signal = get_timing_signal_1d(
            tf.shape(x)[1], tf.shape(x)[2], 
            min_timescale, max_timescale,
            start_index
        )
        return x + signal


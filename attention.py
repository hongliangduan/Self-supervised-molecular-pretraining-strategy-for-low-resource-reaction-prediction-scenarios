#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf


#############################################
##
## multihead attention
##
#############################################
def dot_product_attention(Q, K, V,  
                          attn_bias=None,
                          keep_prob=.9,
                          is_training=True,
                          summaries=None,
                          attn_tensors=None,
                          scope='dot_product_attention'):
    """
    Soft scaled attention mechanism for multihead attention
    """

    # Q: (B, n, T1, d_k), K: (B, n, T2, d_k), V: (B, n, T2, d_v)
    # d_k == d_v
    with tf.variable_scope(scope):
        # get d_k
        # d_k = tf.to_float(tf.shape(Q)[-1])
        d_k = K.get_shape().as_list()[-1]

        # soft attention
        # K_t = tf.transpose(K, [0, 2, 1])
        # scaled: (B, n, T1, d_k) x (B, n, d_k, T2) = (B, n, T1, T2)
        weights = tf.matmul(Q, K, transpose_b=True) / (d_k ** 0.5)

        if attn_bias is not None:
            weights += attn_bias

        # p: (B, n, T1, T2)
        p = tf.nn.softmax(weights, axis=-1)
        if is_training and summaries is not None:
            summaries.append(
                tf.summary.image('atten_p', tf.expand_dims(p[:, 0, :, :], axis=-1), max_outputs=1)
            )
        if attn_tensors is not None:
            attn_tensors.append(p)
            
        if is_training:
            p = tf.nn.dropout(p, keep_prob=keep_prob)

        # (B, n, T1, T2) x (B, n, T2, d_v) = (B, n, T1, d_v)
        return tf.matmul(p, V)


def multihead_attention(Q, K, V,
                        num_units=None, 
                        n_heads=8,
                        use_bias=False,
                        keep_prob=.9,
                        is_training=True,
                        attn_bias=None,
                        summaries=None,
                        attn_tensors=None,
                        scope='multi_head_attention'):
    with tf.variable_scope(scope):
        # Q: (B, T, d_model)
        d_model = Q.get_shape().as_list()[-1]
        if num_units is None:
            num_units = d_model

        # linear projection
        # _Q: (B, T, d_model)  --> (B, T, d_k*n_heads)
        # num_units = d_k * n_heads
        _Q = tf.layers.dense(
            inputs=Q,
            units=num_units,
            use_bias=use_bias,
            name='Q_projection'
        )

        _K = tf.layers.dense(
            inputs=K,
            units=num_units,
            use_bias=use_bias,
            name='K_projection'
        )

        _V = tf.layers.dense(
            inputs=V,
            units=num_units,
            use_bias=use_bias,
            name='V_projection'
        )

        # split to number of heads
        # _Qn: (B, T, d_k*n_heads) --> (B, n_heads, T, d_k)
        _Qn = tf.concat(tf.split(tf.expand_dims(_Q, axis=1), n_heads, axis=-1), axis=1)
        _Kn = tf.concat(tf.split(tf.expand_dims(_K, axis=1), n_heads, axis=-1), axis=1)
        _Vn = tf.concat(tf.split(tf.expand_dims(_V, axis=1), n_heads, axis=-1), axis=1)

        # do attention
        multi_attn = dot_product_attention(_Qn, _Kn, _Vn, 
                                           attn_bias=attn_bias,
                                           keep_prob=keep_prob,
                                           is_training=is_training,
                                           summaries=summaries,
                                           attn_tensors=attn_tensors)

        # concat_outs: (B, n_heads, T, d_k) --> (B, T, d_k*n_heads)
        concat_outs = tf.concat(tf.split(multi_attn, n_heads, axis=1), axis=-1)
        concat_outs = tf.squeeze(concat_outs, axis=1)

        # project back to d_model
        hidd_outs = tf.layers.dense(
            inputs=concat_outs,
            units=d_model,
            use_bias=use_bias,
            name='out_projection'
        )

        return hidd_outs

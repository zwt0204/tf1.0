# -*- encoding: utf-8 -*-
"""
@File    : model_transformer.py
@Time    : 2019/11/25 18:07
@Author  : zwt
@git   : https://www.github.com/kyubyong/transformer.
@Software: PyCharm
"""
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import logging
from utils import convert_idx_to_token_tensor

logging.basicConfig(level=logging.INFO)


class Model_Transformer():

    def __init__(self, batch_size, d_model, num_blocks, num_heads, sequence_length
                 , dropout_rate, smoothing, warmup_steps, vocab_file, d_ff, lr):
        # 批量大小
        self.batch_size = batch_size
        # hidden dimension of encoder/decoder
        self.d_model = d_model
        # 编码层和解码层的个数
        self.num_blocks = num_blocks
        # 多头注意力机制的头个数
        self.num_heads = num_heads
        # 句子长度
        self.sequence_length = sequence_length
        # drop_out
        self.dropout_rate = dropout_rate
        # label smoothing rate 标签平滑率
        self.smoothing = smoothing
        # 当step小于warmup_steps，学习率等于基础学习率×(当前step/warmup_step)
        # 由于后者是一个小于1的数值，因此在整个warm up的过程中，学习率是一个递增的过程！
        # 当warm up结束后，学习率开始递减。
        self.warmup_steps = warmup_steps
        # 词典
        self.vocab_file = vocab_file
        # 词映射
        self.char_index, self.index_char = self.load_dict()
        # 词个数
        self.vocab_size = len(self.char_index)
        # 词嵌入  shape=(self.vocab_size, self.d_model)
        self.embedding = self.get_token_embeddings()
        # hidden dimension of feedforward layer
        self.d_ff = d_ff
        # 学习率
        self.lr = lr

    def load_dict(self):
        """
        返回词映射{word: word_index} and {word_index: wprd}
        :return:
        """
        vocab = [line.split()[0] for line in open(self.vocab_file, 'r', encoding='utf8').read().splitlines()]
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        return token2idx, idx2token

    def get_token_embeddings(self, zero_pad=True):
        """
        获取embedding
        :param zero_pad:
        :return:
        """
        with tf.variable_scope("shared_weight_matrix"):
            # tf.contrib.layers.xavier_initializer()：保持每一层的梯度大小都差不多相同
            embeddings = tf.get_variable('weight_mat',
                                         dtype=tf.float32,
                                         shape=(self.vocab_size, self.d_model),
                                         initializer=tf.contrib.layers.xavier_initializer())
            if zero_pad:
                # tf.concat :向量的拼接
                # 将vocabulary 中index=0的设置为 constant 0, 也就是作为 input 中的 zero padding 的词向量。
                embeddings = tf.concat((tf.zeros(shape=[1, self.d_model]),
                                        embeddings[1:, :]), 0)
        return embeddings

    def encoder(self, xs, training=True):
        """
        编码层
        :param xs:
        :param training:
        :return:
        """
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            x, seqlens, sentsl = xs
            # tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，
            # 否则返回False，返回的值的矩阵维度和A是一样的
            src_mask = tf.math.equal(x, 0)  # [batch_size, embedding_size]
            # embedding
            enc = tf.nn.embedding_lookup(self.embedding, x)  # [batch_size, length, embedding]
            # 归一化，为什么？
            enc *= self.d_model ** 0.5
            enc += self.positional_encoding(enc, self.sequence_length)
            enc = tf.layers.dropout(enc, self.dropout_rate, training=training)
            # Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = self.multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_mask,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = self.ff(enc, num_units=[self.d_ff, self.d_model])
        memory = enc
        return memory, sentsl, src_mask

    def decode(self, ys, memory, src_masks, training=True):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding
            dec = tf.nn.embedding_lookup(self.embedding, decoder_inputs)  # (N, T2, d_model)
            dec *= self.d_model ** 0.5  # scale
            # 位置向量与词向量的融合
            dec += self.positional_encoding(dec, self.sequence_length)
            dec = tf.layers.dropout(dec, self.dropout_rate, training=training)

            # 多层编码
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # 多头注意力机制
                    dec = self.multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # 编码解码多头注意力机制，query为上层输出，key，values为编码层的输出
                    dec = self.multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    # Feed Forward
                    dec = self.ff(dec, num_units=[self.d_ff, self.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embedding)  # (d_model, vocab_size) 转置
        logits = tf.einsum('ntd,dk->ntk', dec, weights)  # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def positional_encoding(self, inputs,
                            maxlen,
                            masking=True,
                            scope="positional_encoding"):
        E = inputs.get_shape().as_list()[-1]  # static
        N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)
            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
                for pos in range(maxlen)])
            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)
            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)
            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
            return tf.to_float(outputs)

    def multihead_attention(self, queries, keys, values, key_masks,
                            num_heads=8,
                            dropout_rate=0,
                            training=True,
                            causality=False,
                            scope="multihead_attention"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections  8个头同时计算
            Q = tf.layers.dense(queries, self.d_model, use_bias=True)  # (N, T_q, d_model)
            K = tf.layers.dense(keys, self.d_model, use_bias=True)  # (N, T_k, d_model)
            V = tf.layers.dense(values, self.d_model, use_bias=True)  # (N, T_k, d_model)
            # Split and concat  分裂成8个头
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)
            # Restore shape  将8个头的结构拼接，每个头的输出刚好是64维，拼接后又是512维
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)
            # 残差连接
            outputs += queries
            # 层归一化
            outputs = self.ln(outputs)

        return outputs

    def ff(self, inputs, num_units, scope="positionwise_feedforward"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])
            # Residual connection
            outputs += inputs
            # Normalize
            outputs = self.ln(outputs)

        return outputs

    def ln(self, inputs, epsilon=1e-8, scope="ln"):
        """
        层归一化
        :param inputs:
        :param epsilon:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]  # self.d_model
            # tf.nn.moments()函数用于计算均值和方差
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def scaled_dot_product_attention(self, Q, K, V, key_masks,
                                     causality=False, dropout_rate=0.,
                                     training=True,
                                     scope="scaled_dot_product_attention"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product 计算score
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
            # scale divide by d_k开根号
            outputs /= d_k ** 0.5
            # key masking
            outputs = self.mask(outputs, key_masks=key_masks, type="key")
            # 如果为true的话，那么就是将这个东西未来的units给屏蔽了
            if causality:
                outputs = self.mask(outputs, type="future")
            # softmax
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
            # dropout
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

    def mask(self, inputs, key_masks=None, type=None):
        """
        inputs: 3d tensor. (N, T_q, T_k)
        keys: 3d tensor. (N, T_k, d)
        :param inputs:
        :param key_masks:
        :param type:
        :return:
        """
        # 填充的负数
        padding_num = -2 ** 32 + 1
        if type in ("k", "key", "keys"):
            # 张量转换为 float32 类型, 将true/false转化为1/0
            key_masks = tf.to_float(key_masks)
            key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
            key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
            outputs = inputs + key_masks * padding_num
        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
            # 转化为下三角矩阵，上三角全为0
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            # 在dim=0 添加一维
            # tf.tile（A， B）将A与B对应的维度扩大响应倍数
            future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)
            # 得到下三角为负数，上三角为0的下三角矩阵
            paddings = tf.ones_like(future_masks) * padding_num
            # tf.equal对应位置相同返回True，否则返回False
            # where(condition, x=None, y=None, name=None)的用法
            # condition， x, y 相同维度，condition是bool型值，True/False
            # 返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素
            outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        else:
            print("Check if you entered type correctly!")

        return outputs

    def train(self, xs, ys):
        # forward

        memory, sents1, src_masks = self.encoder(xs)
        logits, _, y, _ = self.decode(ys, memory, src_masks)

        # train scheme
        y_ = self.label_smoothing(tf.one_hot(y, depth=self.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.char_index["<pad>"]))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        # global_step在滑动平均、优化器、指数衰减学习率等方面都有用到,
        # 代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表
        global_step = tf.train.get_or_create_global_step()
        lr = self.noam_scheme(self.lr, global_step, self.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.char_index["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encoder(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.sequence_length)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.char_index["<pad>"]:
                break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0] - 1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.index_char)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

    def predict(self, xs, ys):
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.char_index["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encoder(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.sequence_length)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.char_index["<pad>"]:
                break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0] - 1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.index_char)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

    def label_smoothing(self, inputs, epsilon=0.1):
        """
        标签平滑
        :param inputs:
        :param epsilon:
        :return:
        """
        V = inputs.get_shape().as_list()[-1]  # self.vocab_size
        return ((1 - epsilon) * inputs) + (epsilon / V)

    def noam_scheme(self, init_lr, global_step, warmup_steps=4000.):
        """
        学习率预热
        :param init_lr:
        :param global_step:
        :param warmup_steps:
        :return:
        """
        # tf.cast:数据类型转换
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
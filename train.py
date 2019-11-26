# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/11/25 18:36
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import tensorflow as tf

from model_transformer import Model_Transformer
import os
import logging
from utils import save_variable_specs, calc_bleu, get_hypotheses, get_batch
import math
from tqdm import tqdm

# GPU设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

logging.basicConfig(level=logging.INFO)


class model_train():

    def __init__(self):
        self.class_graph = tf.Graph()
        self.model_dir = "models/Transformer"
        self.batch_size = 128
        self.d_model = 256
        self.num_blocks = 3
        self.num_heads = 8
        self.sequence_length = 100
        self.dropout_rate = 0.3
        self.smoothing = 0.1
        self.warmup_steps = 400
        self.d_ff = 1024
        self.lr = 0.00001
        self.vocab_file = 'data/iwslt2016/segmented/bpe.vocab'
        self.model = Model_Transformer(self.batch_size, self.d_model, self.num_blocks, self.num_heads,
                                       self.sequence_length
                                       , self.dropout_rate, self.smoothing, self.warmup_steps, self.vocab_file,
                                       self.d_ff, self.lr)
        self.saver = tf.train.Saver()

    def get_batch(self, fpath1, fpath2, maxlen1, maxlen2, shuffle=False):
        sents1, sents2 = self.load_data(fpath1, fpath2, maxlen1, maxlen2)
        batches = self.input_fn(sents1, sents2, shuffle=shuffle)
        num_batches = self.calc_num_batches(len(sents1))
        return batches, num_batches, len(sents1)

    def load_data(self, fpath1, fpath2, maxlen1, maxlen2):
        sents1, sents2 = [], []
        with open(fpath1, 'r', encoding='utf8') as f1, open(fpath2, 'r', encoding='utf8') as f2:
            for sent1, sent2 in zip(f1, f2):
                if len(sent1.split()) + 1 > maxlen1:
                    continue  # 1: </s>
                if len(sent2.split()) + 1 > maxlen2:
                    continue  # 1: </s>
                sents1.append(sent1.strip())
        return sents1, sents2

    def input_fn(self, sents1, sents2, shuffle=False):
        shapes = (([None], (), ()),
                  ([None], [None], (), ()))
        types = ((tf.int32, tf.int32, tf.string),
                 (tf.int32, tf.int32, tf.int32, tf.string))
        paddings = ((0, 0, ''),
                    (0, 0, 0, ''))

        dataset = tf.data.Dataset.from_generator(
            generator_fn,
            output_shapes=shapes,
            output_types=types,
            args=(sents1, sents2))

        if shuffle:  # for training
            dataset = dataset.shuffle(128 * self.batch_size)

        dataset = dataset.repeat()  # iterate forever
        dataset = dataset.padded_batch(self.batch_size, shapes, paddings).prefetch(1)

        return dataset

    def calc_num_batches(self, total_num):
        return total_num // self.batch_size + int(total_num % self.batch_size != 0)

    def train_(self, epochs):
        train_batches, num_train_batches, num_train_samples = get_batch('data/iwslt2016/segmented/train.de.bpe',
                                                                        'data/iwslt2016/segmented/train.en.bpe',
                                                                        self.sequence_length,
                                                                        self.sequence_length,
                                                                        self.vocab_file, self.batch_size,
                                                                        shuffle=True)
        eval_batches, num_eval_batches, num_eval_samples = get_batch('data/iwslt2016/segmented/eval.de.bpe',
                                                                     'data/iwslt2016/segmented/eval.en.bpe',
                                                                     100000, 100000,
                                                                     self.vocab_file, self.batch_size,
                                                                     shuffle=False)
        iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
        xs, ys = iter.get_next()

        train_init_op = iter.make_initializer(train_batches)
        eval_init_op = iter.make_initializer(eval_batches)
        loss, train_op, global_step, train_summaries = self.model.train(xs, ys)
        y_hat, eval_summaries = self.model.eval(xs, ys)

        logging.info("# Session")
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.model_dir)
            if ckpt is None:
                logging.info("Initializing from scratch")
                sess.run(tf.global_variables_initializer())
                save_variable_specs(os.path.join('data/log/1', "specs"))
            else:
                self.saver.restore(sess, ckpt)

            summary_writer = tf.summary.FileWriter(self.model_dir, sess.graph)

            sess.run(train_init_op)
            total_steps = epochs * num_train_batches
            _gs = sess.run(global_step)
            for i in tqdm(range(_gs, total_steps + 1)):
                _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
                epoch = math.ceil(_gs / num_train_batches)
                summary_writer.add_summary(_summary, _gs)

                if _gs and _gs % num_train_batches == 0:
                    logging.info("epoch {} is done".format(epoch))
                    _loss = sess.run(loss)  # train loss

                    logging.info("# test evaluation")
                    _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
                    summary_writer.add_summary(_eval_summaries, _gs)

                    logging.info("# get hypotheses")
                    hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, self.model.index_char)
                    print('====',hypotheses)
                    logging.info("# write results")
                    model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
                    if not os.path.exists('data/eval/1'):
                        os.makedirs('data/eval/1')
                    translation = os.path.join('data/eval/1', model_output)
                    with open(translation, 'w') as fout:
                        fout.write("\n".join(hypotheses))

                    logging.info("# calc bleu score and append it to translation")
                    calc_bleu('data/iwslt2016/prepro/eval.en', translation)

                    logging.info("# save models")
                    self.saver.save(sess, os.path.join(self.model_dir, 'transformer.dat'), global_step=_gs)
                    sess.run(train_init_op)
            summary_writer.close()

        logging.info("Done")


def generator_fn(sents1, sents2):
    token2idx, _ = load_vocab('data/iwslt2016/segmented/bpe.vocab')
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)


def encode(inp, type, dict):
    inp_str = inp.decode("utf-8")
    if type == "x":
        tokens = inp_str.split() + ["</s>"]
    else:
        tokens = ["<s>"] + inp_str.split() + ["</s>"]

    x = [dict.get(t, dict["<unk>"]) for t in tokens]
    return x


def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r', encoding='utf8').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token


if __name__ == '__main__':
    train_model = model_train()
    train_model.train_(1)

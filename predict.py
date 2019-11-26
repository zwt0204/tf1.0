# -*- encoding: utf-8 -*-
"""
@File    : predict.py
@Time    : 2019/11/26 10:37
@Author  : zwt
@git   : 
@Software: PyCharm
"""
from model_transformer import Model_Transformer
import tensorflow as tf
from utils import get_hypotheses


class Model_Predict():

    def __init__(self, **kwargs):
        self.model_dir = kwargs["model_dir"]
        self.graph = kwargs["graph"]
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
        self.vocab_file = kwargs["vocab_file"]
        self.model = Model_Transformer(self.batch_size, self.d_model, self.num_blocks, self.num_heads,
                                       self.sequence_length
                                       , self.dropout_rate, self.smoothing, self.warmup_steps, self.vocab_file,
                                       self.d_ff, self.lr)
        self.saver = tf.train.Saver()
        config = tf.ConfigProto(log_device_placement=False)
        self.session = tf.Session(graph=self.graph, config=config)
        self.load()

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt is not None and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            raise Exception("load classification failure...")

    def predict(self, input_text):
        input_text = input_text
        test_batches, num_test_samples = self.get_batch(input_text, self.batch_size,
                                                        shuffle=False)
        iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
        xs = iter.get_next()
        test_init_op = iter.make_initializer(test_batches)
        pred, y_hat, summaries = self.model.predict(xs)
        self.session.run(test_init_op)
        hypotheses = get_hypotheses(1, num_test_samples, self.session.run(y_hat), y_hat, self.model.index_char)
        print(hypotheses)

    def get_batch(self, data, batch_size, shuffle=False):
        sents1 = self.load_data(data)
        batches = self.input_fn(sents1, batch_size, shuffle=shuffle)
        return batches, len(sents1)

    def load_data(self, data):
        sents1 = []
        for sent1 in data:
            if len(sent1.split()) + 1 > self.sequence_length:
                continue  # 1: </s>
            sents1.append('▁'.strip() + sent1.strip())
        return sents1

    def generator_fn(self, sents1):
        for sent1 in sents1:
            x = self.encode(sent1, "x", self.model.char_index)

            x_seqlen = len(x)
            yield (x, x_seqlen, sent1)

    def input_fn(self, sents1, batch_size, shuffle=False):
        shapes = (([None], (), ()),
                  ([None], [None], (), ()))
        types = ((tf.int32, tf.int32, tf.string),
                 (tf.int32, tf.int32, tf.int32, tf.string))
        paddings = ((0, 0, ''),
                    (0, 0, 0, ''))

        dataset = tf.data.Dataset.from_generator(
            self.generator_fn,
            output_shapes=shapes,
            output_types=types,
            args=(sents1))  # <- arguments for generator_fn. converted to np string arrays

        if shuffle:  # for training
            dataset = dataset.shuffle(128 * batch_size)

        dataset = dataset.repeat()  # iterate forever
        dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

        return dataset

    def encode(self, inp, type, dict):
        inp_str = inp.decode("utf-8")
        if type == "x":
            tokens = inp_str.split() + ["</s>"]
        else:
            tokens = ["<s>"] + inp_str.split() + ["</s>"]

        x = [dict.get(t, dict["<unk>"]) for t in tokens]
        return x


if __name__ == '__main__':
    graph = tf.Graph()
    model_dir = "models/Transformer"
    vocab_file = 'data\iwslt2016\segmented\\bpe.vocab'
    Predict_model = Model_Predict(model_dir=model_dir, graph=graph, vocab_file=vocab_file)
    Predict_model.predict('Hello')
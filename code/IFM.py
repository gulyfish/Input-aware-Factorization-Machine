import math
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import pickle as pk
from sklearn.metrics import mean_absolute_error

#################### Arguments ####################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('process', 'train','Process type: train, evaluate.')
tf.app.flags.DEFINE_string('path', 'attentional_factorization_machine-master/data/','Input data path.')
tf.app.flags.DEFINE_string('dataset', 'frappe', 'Choose a dataset.')
tf.app.flags.DEFINE_integer('epoch', 100, 'Number of epochs.')
tf.app.flags.DEFINE_string("metric", "MAE", " MAE OR RMSE")
tf.app.flags.DEFINE_integer('pretrain', 1, 'flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save to pretrain file')
tf.app.flags.DEFINE_integer('batch_size', 2048, 'Batch size.')
tf.app.flags.DEFINE_integer('embedding_size', 256, 'Number of embedding size.')
tf.app.flags.DEFINE_integer('valid_dimen', 10, 'Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3)')
tf.app.flags.DEFINE_string('fenlayers', '[512,258,128]', 'Size of each layer in FEN.')
tf.app.flags.DEFINE_float("lamda", 0.01, "Regularizer for FEN part")
tf.app.flags.DEFINE_string('keep', '[0.8,0.7,0.7,0.8]', 'Keep probility (1-dropout) for the bilinear interaction layer. 1: no dropout')
tf.app.flags.DEFINE_float('lr', 0.01, 'Learning rate.')
tf.app.flags.DEFINE_string('optimizer', 'AdagradOptimizer', 'Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
tf.app.flags.DEFINE_integer('verbose', 1, 'Whether to show the performance of each epoch (0 or 1)')
tf.app.flags.DEFINE_integer('batch_norm', 1, 'Whether to perform batch normaization (0 or 1)')
tf.app.flags.DEFINE_bool('freeze_embedding', True, 'Freese all params of fm and learn attention params only')


class FM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, pretrain_flag, save_file, embedding_size, valid_dimension, epoch, metric_type,batch_size,
                 learning_rate, lamda, keep, fenlayers, freeze_embedding,optimizer_type, batch_norm, verbose, random_seed=2018):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.valid_dimension = valid_dimension
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.lamda = lamda
        self.keep = np.array(keep)
        self.fenlayers = fenlayers
        self.freeze_embedding = freeze_embedding
        self.epoch = epoch
        self.metric_type = metric_type
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None],name="train_features_fm")  # None * features_M
            # self.memory_features = tf.placeholder(tf.int32, shape=[None, None],name="memory_features")  # None * features_M+1
            # self.nomemory_features = tf.placeholder(tf.int32, shape=[None, None],name="memory_features")  # None * features_M+1
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels_fm")  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep_fm")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase_fm")

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # get the summed up embeddings of features.
            self.nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features,
                                                             name='nonzero_embeddings')


            # Factor Estimating Net
            dnn_nonzero_embeddings = tf.reshape(self.nonzero_embeddings,shape=[-1, self.valid_dimension * self.embedding_size])
            self.dnn = tf.add(tf.matmul(dnn_nonzero_embeddings, self.weights['fenlayer_0']),self.weights['fenbias_0'])  # None * layer[i] * 1
            if self.batch_norm:
                self.dnn = self.batch_norm_layer(self.dnn, train_phase=self.train_phase,scope_bn='bn_0')  # None * layer[i] * 1
            self.dnn = tf.nn.relu(self.dnn)
            self.dnn = tf.nn.dropout(self.dnn, self.dropout_keep[0])  # dropout at each Deep layer
            for i in range(1, len(self.fenlayers)):
                self.dnn = tf.add(tf.matmul(self.dnn, self.weights['fenlayer_%d' % i]),self.weights['fenbias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.dnn = self.batch_norm_layer(self.dnn, train_phase=self.train_phase,scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.dnn = tf.nn.relu(self.dnn)
                self.dnn = tf.nn.dropout(self.dnn, self.dropout_keep[i])  # dropout at each Deep layer
            self.dnn_out = tf.matmul(self.dnn, self.weights['prediction_dnn'])  # None * 10
            self.outm = tf.constant(float(self.valid_dimension)) * tf.nn.softmax(self.dnn_out)
            # self.dnn_out = tf.matmul(self.dnn_out, self.weights['prediction'])

            self.nonzero_embeddings_m = tf.multiply(self.nonzero_embeddings, tf.expand_dims(self.outm, 2))


            # FM Prediction Layer
            element_wise_product_list = []
            count = 0
            for i in range(0, self.valid_dimension):
                for j in range(i + 1, self.valid_dimension):
                    element_wise_product_list.append(
                        tf.multiply(self.nonzero_embeddings_m[:, i, :], self.nonzero_embeddings_m[:, j, :]))
                    count += 1
            self.element_wise_product = tf.stack(element_wise_product_list)  # (M'*(M'-1)) * None * K
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2],
                                                     name="element_wise_product")  # None * (M'*(M'-1)) * K
            self.interactions = tf.reduce_sum(self.element_wise_product, 2, name="interactions")

            self.IFM = tf.reduce_sum(self.element_wise_product, 1, name="ifm")
            self.IFM = tf.nn.dropout(self.IFM, self.dropout_keep[-1])

            # _________out _________

            self.Bilinear = tf.reduce_sum(self.IFM, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features),
                            tf.expand_dims(self.outm, 2)), 1)  # None * 1
            self.Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([self.Bilinear, self.Feature_bias, self.Bias],name="out")  # None * 1

            # Compute the square loss.
            if self.lamda > 0:
                keys = list(self.weights.keys())
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.add_n(
                    [tf.contrib.layers.l2_regularizer(self.lamda)(self.weights[v]) for v in self.weights])
                #           + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_memory'])# regulizer
                # + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings']) self.l2_embed * tf.add_n([tf.nn.l2_loss(v) for v in embeds])
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.sess = self._init_session()
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess.run(init)

            # number of params
            # total_parameters = 0
            # for variable in self.weights.values():
            #     shape = variable.get_shape()  # shape is an array of tf.Dimension
            #     variable_parameters = 1
            #     for dim in shape:
            #         variable_parameters *= dim.value
            #     total_parameters += variable_parameters
            # if self.verbose > 0:
            #     print("#params: %d" % total_parameters)

    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()
        trainable = self.freeze_embedding
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')

            with self._init_session() as sess:
                weight_saver.restore(sess, self.save_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])

            weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32, name='feature_embeddings',
                                                            trainable=trainable)
            weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32, name='feature_bias', trainable=trainable)
            weights['bias'] = tf.Variable(b, dtype=tf.float32, name='bias', trainable=trainable)
        else:
            weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.embedding_size], 0.0, 0.01),
                name='feature_embeddings')  # features_M * K
            weights['feature_bias'] = tf.Variable(
                tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1
            weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1




        num_fenlayer = len(self.fenlayers)
        if num_fenlayer > 0:
            glorot = np.sqrt(2.0 / (self.embedding_size + self.fenlayers[0]))

            weights['fenlayer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.valid_dimension * self.embedding_size, self.fenlayers[0])),
                dtype=np.float32)
            weights['fenbias_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.fenlayers[0])),
                dtype=np.float32)  # 1 * layers[0]

            for i in range(1, num_fenlayer):
                glorot = np.sqrt(2.0 / (self.fenlayers[i - 1] + self.fenlayers[i]))
                weights['fenlayer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.fenlayers[i - 1], self.fenlayers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights['fenbias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.fenlayers[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.fenlayers[-1] + 1))

            weights['prediction_dnn'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.fenlayers[-1], self.valid_dimension)),dtype=np.float32)  # layers[-1] * 1

        return  weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def layer_norm_layer(self, x, train_phase, scope_bn):
        bn_train = tf.contrib.layers.layer_norm( x,center=True,scale=True,activation_fn=None,reuse=tf.AUTO_REUSE,
        variables_collections=None,outputs_collections=None,trainable=True,begin_norm_axis=1,begin_params_axis=-1,scope=scope_bn)
        bn_inference = tf.contrib.layers.layer_norm( x,center=True,scale=True,activation_fn=None,reuse=tf.AUTO_REUSE,
        variables_collections=None,outputs_collections=None,trainable=False,begin_norm_axis=1,begin_params_axis=-1,scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y, KEY, NOKEY = [], [], [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b):  # shuffle two lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, Train_data, Validation_data, Test_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            print("Init: \t train=%.4f, validation=%.4f [%.1f s]" % (init_train, init_valid, time() - t2))

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # evaluate training and validation datasets
            train_result = self.evaluate(Train_data)
            valid_result = self.evaluate(Validation_data)
            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain=%.4f, validation=%.4f [%.1f s]"
                      % (epoch + 1, t2 - t1, train_result, valid_result, time() - t2))

            test_result = self.evaluate(Test_data)
            print("Epoch %d [%.1f s]\ttest=%.4f [%.1f s]"
                  % (epoch + 1, t2 - t1, test_result, time() - t2))
            if self.eva_termination(self.valid_rmse):
                break

        if self.pretrain_flag < 0 or self.pretrain_flag == 2:
            print("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)

    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def get_ordered_block_from_data(self, data, batch_size, index):  # generate a ordered block of data
        start_index = index * batch_size
        X, Y, KEY, NOKEY = [], [], [], []
        # get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append(data['Y'][i])

                X.append(data['X'][i])
                i = i + 1
            else:
                break

        return {'X': X, 'Y': Y}

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        # fetch the first batch
        batch_index = 0
        batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)
        # batch_xs = data
        y_pred, A, B, C, D = None, None, None, None, None
        # if len(batch_xs['X']) > 0:
        while len(batch_xs['X']) > 0:
            num_batch = len(batch_xs['Y'])
            # [[y] for y in batch_xs['Y']]
            feed_dict = {self.train_features: batch_xs['X'],
                         self.train_labels: [[y] for y in batch_xs['Y']],
                         self.dropout_keep: list(1.0 for i in range(len(self.keep))), self.train_phase: False}
            batch_out = self.sess.run((self.out ), feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)
        y_true = np.reshape(data['Y'], (num_example,))

        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded,
                                         np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        MAE=mean_absolute_error(y_true, predictions_bounded)
        if self.metric_type == 'MAE':
            return MAE
        else:
            return RMSE


def make_save_file(FLAGS):
    pretrain_path = 'attentional_factorization_machine-master/pretrain/fm_%s_%d' % (FLAGS.dataset, FLAGS.embedding_size)

    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    save_file = pretrain_path + '/%s_%d' % (FLAGS.dataset, FLAGS.embedding_size)
    return save_file


def train(FLAGS):
    # Data loading
    data = DATA.LoadData(FLAGS.path, FLAGS.dataset)

    if FLAGS.verbose > 0:
        print(
            "IFM: dataset=%s, embedding_size=%d,#epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%s, metric=%s, optimizer=%s, batch_norm=%d"
            % (
                FLAGS.dataset, FLAGS.embedding_size, FLAGS.epoch, FLAGS.batch_size, FLAGS.lr, FLAGS.lamda, FLAGS.keep, FLAGS.metric,
                FLAGS.optimizer, FLAGS.batch_norm))

    # Training
    t1 = time()
    model = FM(data.features_M, FLAGS.pretrain, make_save_file(FLAGS),  FLAGS.embedding_size, FLAGS.valid_dimen,
               FLAGS.epoch, FLAGS.metric,FLAGS.batch_size, FLAGS.lr, FLAGS.lamda, eval(FLAGS.keep),
               eval(FLAGS.fenlayers), FLAGS.freeze_embedding, FLAGS.optimizer, FLAGS.batch_norm, FLAGS.verbose)
    model.train(data.Train_data, data.Validation_data, data.Test_data)

    # Find the best validation result across iterations
    best_valid_score = 0
    best_valid_score = min(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print("Best Iter(validation)= %d\t train = %.4f, valid = %.4f [%.1f s]"
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time() - t1))


def evaluate(FLAGS):
    # load test data
    data = DATA.LoadData(FLAGS.path, FLAGS.dataset).Test_data
    save_file = make_save_file(FLAGS)

    # load the graph
    weight_saver = tf.train.import_meta_graph(save_file + '.meta')
    pretrain_graph = tf.get_default_graph()

    # load tensors
    feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
    nonzero_embeddings = pretrain_graph.get_tensor_by_name('nonzero_embeddings:0')
    feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
    bias = pretrain_graph.get_tensor_by_name('bias:0')
    fm = pretrain_graph.get_tensor_by_name('fm:0')
    fm_out = pretrain_graph.get_tensor_by_name('fm_out:0')
    out = pretrain_graph.get_tensor_by_name('out:0')
    train_features = pretrain_graph.get_tensor_by_name('train_features_fm:0')
    train_labels = pretrain_graph.get_tensor_by_name('train_labels_fm:0')
    dropout_keep = pretrain_graph.get_tensor_by_name('dropout_keep_fm:0')
    train_phase = pretrain_graph.get_tensor_by_name('train_phase_fm:0')

    # restore session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    weight_saver.restore(sess, save_file)

    # start evaluation
    num_example = len(data['Y'])
    feed_dict = {train_features: data['X'], train_labels: [[y] for y in data['Y']], dropout_keep: 1.0,
                 train_phase: False}
    ne, fe = sess.run((nonzero_embeddings, feature_embeddings), feed_dict=feed_dict)
    _fm, _fm_out, predictions = sess.run((fm, fm_out, out), feed_dict=feed_dict)

    # calculate rmse
    y_pred = np.reshape(predictions, (num_example,))
    y_true = np.reshape(data['Y'], (num_example,))

    predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
    predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
    RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))

    print("Test RMSE: %.4f" % (RMSE))

def main(_):
    if FLAGS.process == 'train':
        train(FLAGS)
    elif FLAGS.process == 'evaluate':
        evaluate(FLAGS)

if __name__ == '__main__':
    tf.app.run()

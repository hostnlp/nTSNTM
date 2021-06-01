from __future__ import print_function

import heapq
import numpy as np
import tensorflow as tf
import math
import utils
import os,sys
sys.path.append('utils')

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data/reuters', 'Data dir path.')
flags.DEFINE_string('dataset','reuters', 'Data dir path.')
flags.DEFINE_integer('vocab_size', 2000, 'Vocabulary size.')
flags.DEFINE_float('learning_rate', 5e-4, 'Learning rate.')
flags.DEFINE_float('decay_rate', 0.03, 'annealing rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('prior_alpha', 1., 'prior alpha.')
flags.DEFINE_float('prior_beta', 10., 'prior beta.')
flags.DEFINE_integer('n_epoch', 100, 'Size of stochastic vector.')
flags.DEFINE_integer('n_hidden', 256, 'Size of each hidden layer.')
flags.DEFINE_integer('truncation_level', 200, 'Size of stochastic vector.')
flags.DEFINE_integer('n_topic2', 30, 'Size of stochastic vector.')
flags.DEFINE_boolean('test', True, 'Process test data.')
flags.DEFINE_string('non_linearity', 'sigmoid', 'Non-linearity of the MLP.')
FLAGS = flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class nTSNTM(object):

    def __init__(self,
                 prior_alpha,
                 prior_beta,
                 vocab_size,
                 n_hidden,
                 truncation_level,
                 n_topic2,
                 learning_rate, 
                 batch_size,
                 non_linearity):
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = truncation_level
        self.n_topic2 = n_topic2
        self.n_topic3 = 1
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        self.mask = tf.placeholder(tf.float32, [None], name='mask')  # mask paddings
        self.decay = tf.placeholder(tf.float32, name='decay')

        # encoder
        with tf.variable_scope('encoder'):
          self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity)
          self.a = tf.nn.softplus(utils.small_linear(self.enc_vec, self.n_topic, scope='a'))
          self.b = tf.nn.softplus(utils.small_linear(self.enc_vec, self.n_topic, scope='b'))
          self.mean = utils.linear(self.enc_vec, self.n_hidden, scope='mean')
          self.logsigm = utils.linear(self.enc_vec,
                                     self.n_hidden,
                                     bias_start_zero=True,
                                     matrix_start_zero=True,
                                     scope='logsigm')
          self.kld_gauss = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.logsigm - tf.exp(2 * self.logsigm), 1)
          self.kld_gauss = (self.mask*self.kld_gauss)  # mask paddings

          kl = 1. / (1 + self.a * self.b) * self.Beta_fn(1. / self.a, self.b)
          kl += 1. / (2 + self.a * self.b) * self.Beta_fn(2. / self.a, self.b)
          kl += 1. / (3 + self.a * self.b) * self.Beta_fn(3. / self.a, self.b)
          kl += 1. / (4 + self.a * self.b) * self.Beta_fn(4. / self.a, self.b)
          kl += 1. / (5 + self.a * self.b) * self.Beta_fn(5. / self.a, self.b)
          kl += 1. / (6 + self.a * self.b) * self.Beta_fn(6. / self.a, self.b)
          kl += 1. / (7 + self.a * self.b) * self.Beta_fn(7. / self.a, self.b)
          kl += 1. / (8 + self.a * self.b) * self.Beta_fn(8. / self.a, self.b)
          kl += 1. / (9 + self.a * self.b) * self.Beta_fn(9. / self.a, self.b)
          kl += 1. / (10 + self.a * self.b) * self.Beta_fn(10. / self.a, self.b)
          kl *= (prior_beta - 1) * self.b

          psi_b_taylor_approx = tf.digamma(self.b)
          kl += (self.a - prior_alpha) / self.a * (
                      -0.57721 - psi_b_taylor_approx - 1 / self.b)  # T.psi(self.posterior_b)

          # add normalization constants
          kl += tf.log(self.a * self.b) + tf.log(self.Beta_fn(prior_alpha, prior_beta))

          # final term
          kl += -(self.b - 1) / self.b
          self.kld_ku = tf.reduce_sum(kl, 1)
          self.kld_ku = (self.mask*self.kld_ku)  # mask paddings
          self.kld = (self.kld_ku + self.kld_gauss)

        self.remaining_stick = tf.Variable(tf.ones(shape=[self.batch_size], dtype=tf.float32))
        index = tf.range(0, self.n_topic)
        one_hot = tf.one_hot(index,self.n_topic)
        a = tf.ones(shape=[self.batch_size,self.n_topic,self.n_topic], dtype=tf.float32)
        self.mat1 = a - one_hot
        self.mat2 = tf.linalg.band_part(a, 0, -1)
        self.mat3 = tf.linalg.band_part(a, self.n_topic, 0) -one_hot

        with tf.variable_scope('decoder'):
            eps0 = tf.random_normal((batch_size, self.n_hidden), 0, 1)
            doc_vec = tf.multiply(tf.exp(self.logsigm), eps0) + self.mean
            self.eta = tf.nn.softmax(utils.mlp(doc_vec, [self.n_hidden, 3], self.non_linearity, scope='weight'),
                                        axis=1)
            self.eps = tf.random_uniform((batch_size, self.n_topic), 0.01, 0.99)
            self.v_samples = (1 - (self.eps ** (1 / self.b))) ** (1 / self.a)
            v_mid1 = tf.reshape(self.v_samples, [self.batch_size, self.n_topic, 1])
            self.v_mid2 = self.mat1 - tf.tile(v_mid1, (1, 1, self.n_topic))
            self.v_mid3 = tf.multiply(self.mat2, self.v_mid2) + self.mat3
            self.stick_segment = -tf.reduce_prod(self.v_mid3, 1)
            self.remaining_stick = tf.div(self.stick_segment, self.v_samples)
            self.theta = self.stick_segment

            word_vec = tf.Variable(tf.glorot_uniform_initializer()((self.vocab_size, self.n_hidden)))
            topic_vec = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic, self.n_hidden)))
            temperature = 0.05/(self.decay+1e-5)
            beta_mat = tf.matmul(topic_vec,word_vec ,transpose_b=True)
            self.beta =  tf.nn.softmax(beta_mat, axis=1)

            self.depend = tf.nn.softmax(tf.Variable(tf.glorot_uniform_initializer()((self.n_topic, self.n_topic2)))/temperature)
            self.theta2 = tf.matmul(self.theta, self.depend)
            topic_vec2 = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic2, self.n_hidden)))
            temperature = 1**0.5
            beta_mat2 = tf.matmul(topic_vec2,word_vec ,transpose_b=True)/temperature
            self.beta2 = tf.nn.softmax(beta_mat2, axis=1)
            self.logits2 = tf.matmul(self.theta2, self.beta2, transpose_b=False)
            self.recons_loss2 = -tf.reduce_sum(tf.multiply(self.logits2, self.x), 1)

            self.depend2 = tf.nn.softmax(
                tf.Variable(tf.glorot_uniform_initializer()((self.n_topic2, self.n_topic3))),
                axis=1)
            self.theta3 = tf.matmul(self.theta2, self.depend2)
            topic_vec3 = tf.Variable(tf.glorot_uniform_initializer()((self.n_topic3, self.n_hidden)))
            temperature = 1**(1)
            beta_mat3 = tf.matmul(topic_vec3,word_vec ,transpose_b=True)/temperature
            self.beta3 = tf.nn.softmax(beta_mat3, axis=1)
            self.logits3 = tf.matmul(self.theta3, self.beta3, transpose_b=False)
            self.recons_loss3 = -tf.reduce_sum(tf.multiply(self.logits3, self.x), 1)

            self.logits = tf.matmul(self.theta, self.beta)
            self.final_logits = tf.multiply( tf.transpose(self.eta[:,0]) , tf.transpose(self.logits)) + \
                                tf.multiply( tf.transpose(self.eta[:,1]) , tf.transpose(self.logits2))+ \
                                tf.multiply(tf.transpose(self.eta[:, 2]), tf.transpose(self.logits3))
            self.final_logits = tf.log(tf.transpose(self.final_logits))

        self.recons_loss = -tf.reduce_sum(tf.multiply(self.final_logits, self.x), 1)
        self.objective = self.recons_loss + self.decay*self.kld

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        fullvars = tf.trainable_variables()
        enc_vars = utils.variable_parser(fullvars, 'encoder')
        dec_vars = utils.variable_parser(fullvars, 'decoder')
        enc_grads = tf.gradients(self.objective, enc_vars)
        dec_grads = tf.gradients(self.objective, dec_vars)
        self.optim_full = optimizer.apply_gradients(list(zip(enc_grads, enc_vars)) + list(zip(dec_grads, dec_vars)))

    def Beta_fn(self, a, b):
        return tf.exp(tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a + b))

def train(sess, model, 
          train_url, 
          test_url, 
          batch_size,
          training_epochs=FLAGS.n_epoch,
          alternate_epochs=10):
  """train nTSNTM model."""
  train_set0, train_mat,train_count0 = utils.data_set(train_url,model.vocab_size )
  test_set, test_mat,test_count = utils.data_set(test_url,model.vocab_size)
  dev_size = int(len(train_set0)/20)
  train_size = len(train_set0)-dev_size
  train_set = train_set0[:train_size]
  dev_set = train_set0[train_size:]
  train_count = train_count0[:train_size]
  dev_count = train_count0[train_size:]
  data_mat = np.concatenate((train_mat,test_mat))

  vocab = []
  with open(FLAGS.data_dir + '/' + FLAGS.dataset + '.vocab', 'r') as file_to_read:
      while True:
          lines = file_to_read.readline()
          if not lines:
              break
          word, num = lines.split()  #
          vocab.append(word)  #

  dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False) #
  test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)
  corpus_bow = np.sum(data_mat, 0)
  corpus_topic = corpus_bow / np.linalg.norm(corpus_bow)

  effective_dims_bool = np.ones(FLAGS.truncation_level)
  effective_dims_bool2 = np.ones(FLAGS.n_topic2)

  for epoch in range(1,training_epochs):
    train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
    #-------------------------------
    # train
    decay = min(1.,epoch*FLAGS.decay_rate)
    optim = model.optim_full
    print_mode = 'updating'

    for i in range(alternate_epochs):
        loss_sum = 0.0
        ppx_sum = 0.0
        kld_sum = 0.0
        NLL_sum = 0.0
        word_count = 0
        doc_count = 0
        ratio_list = np.zeros([FLAGS.truncation_level])
        for idx_batch in train_batches:
            data_batch, count_batch, mask = utils.fetch_data(
                train_set, train_count, idx_batch,
                FLAGS.vocab_size)  #
            input_feed = {model.x.name: data_batch, model.mask.name: mask,model.decay:decay}

            _, (all_loss, recon_loss,kld,theta) = sess.run((optim,
                                       [model.objective,model.recons_loss, model.kld,model.theta]),
                                      input_feed)  # loss: (batch_size)
            for s in range(FLAGS.batch_size):
                ratio_list += theta[s]
            loss = (recon_loss+kld)*mask
            loss_sum += np.sum(loss)

            kld_sum += np.sum(kld) / np.sum(mask)
            word_count += np.sum(count_batch)
            count_batch = np.add(count_batch, 1e-12)

            NLL_sum += np.sum(np.divide(recon_loss, count_batch))
            ppx_sum += np.sum(np.divide(loss, count_batch))
            doc_count += np.sum(mask)
        ratio_list /= np.sum(ratio_list)
        cdf = []
        cur = 1
        for s in ratio_list:
            cur -= s
            cdf.append(cur)
        effective_dims_bool =  np.array(cdf) > 0.05
        effective_dims = np.sum(effective_dims_bool)
        #Computing the tree dependency
        depend = sess.run((model.depend),{model.decay:0.1}) > 0.5
        effective_dims_bool2 = (np.inner(np.transpose(depend),effective_dims_bool))>0
        print_ppx = np.exp(loss_sum / word_count)
        print_ppx_perdoc = np.exp(ppx_sum / doc_count)
        print_NLL_perdoc = np.exp(NLL_sum/ doc_count)
        print_kld = kld_sum / len(train_batches)

        print('| Epoch train: {:d} |'.format(epoch ),
              print_mode, '{:d}'.format(i),
              '| Corpus ppx: {:.5f}'.format(print_ppx),  # perplexity for all docs
              '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),  # perplexity for per doc
              '| Per doc NLL: {:.5f}'.format(print_NLL_perdoc),  # perplexity for per doc
              '| KLD: {:.5}'.format(print_kld),
              '| effective_dims: {:d}'.format(effective_dims))
    #-------------------------------
    # dev
    loss_sum = 0.0
    kld_sum = 0.0
    ppx_sum = 0.0
    word_count = 0
    doc_count = 0
    for idx_batch in dev_batches:
      data_batch, count_batch, mask = utils.fetch_data(
          dev_set, dev_count, idx_batch, FLAGS.vocab_size)
      input_feed = {model.x.name: data_batch, model.mask.name: mask,model.decay:1.0}
      loss, kld = sess.run([model.objective, model.kld],
                           input_feed)
      recon_loss = sess.run(model.recons_loss,
                            input_feed)
      loss = recon_loss + kld
      loss_sum += np.sum(loss)
      kld_sum += np.sum(kld) / np.sum(mask)  
      word_count += np.sum(count_batch)
      count_batch = np.add(count_batch, 1e-12)
      ppx_sum += np.sum(np.divide(loss, count_batch))
      doc_count += np.sum(mask) 
    print_ppx = np.exp(loss_sum / word_count)
    print_ppx_perdoc = np.exp(ppx_sum / doc_count)
    print_kld = kld_sum/len(dev_batches)
    print('| Epoch dev: {:d} |'.format(epoch),
           '| Perplexity: {:.9f}'.format(print_ppx),
           '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
           '| KLD: {:.5}'.format(print_kld))        
    #-------------------------------
    # test
    if FLAGS.test:
      loss_sum = 0.0
      kld_sum = 0.0
      ppx_sum = 0.0
      word_count = 0
      doc_count = 0
      NLL_sum = 0
      input_feed = []
      for idx_batch in test_batches:
        data_batch, count_batch, mask = utils.fetch_data(
          test_set, test_count, idx_batch, FLAGS.vocab_size)
        input_feed = {model.x.name: data_batch, model.mask.name: mask,model.decay:1.0}
        loss, kld = sess.run([model.objective, model.kld],input_feed)
        recon_loss = sess.run(model.recons_loss,input_feed)
        loss = recon_loss+kld
        loss_sum += np.sum(loss)
        kld_sum += np.sum(kld)/np.sum(mask) 
        word_count += np.sum(count_batch)
        count_batch = np.add(count_batch, 1e-12)
        ppx_sum += np.sum(np.divide(loss, count_batch))
        NLL_sum += np.sum(np.divide(recon_loss, count_batch))
        doc_count += np.sum(mask) 
      print_ppx = np.exp(loss_sum / word_count)
      print_ppx_perdoc = np.exp(ppx_sum / doc_count)
      print_NLL_perdoc = np.exp(NLL_sum / doc_count)
      print_kld = kld_sum/len(test_batches)
      print('| Epoch test: {:d} |'.format(epoch),
             '| Perplexity: {:.9f}'.format(print_ppx),
             '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
            '| Per doc NLL: {:.5f}'.format(print_NLL_perdoc),
             '| KLD: {:.5}\n'.format(print_kld) )
    top_list = [5,10,15]
    if (epoch+1) % 5 == 0:
        tree = build_tree(sess, model, [effective_dims_bool, effective_dims_bool, effective_dims_bool2])
        print_tree(tree, vocab)
        beta_list = [[],[],[]]
        get_topics(tree, beta_list)
        beta1 = np.array(beta_list[0])
        beta2 = np.array(beta_list[1])
        beta3 = np.array(beta_list[2])
        beta0 = np.concatenate((beta1, np.concatenate((beta2,beta3),axis = 0)), axis=0)
        print('npmi:'+str(utils.evaluate_coherence(beta0, data_mat, top_list)))
        print('TU:'+str(utils.evaluate_TU(beta0, top_list)))
        topics_specs3 = utils.compute_topic_specialization(beta3, corpus_topic)
        topics_specs2 = utils.compute_topic_specialization(beta2, corpus_topic)
        topics_specs1 = utils.compute_topic_specialization(beta1, corpus_topic)
        print('level 1 topic specialization: ' + str(topics_specs1))
        print('level 2 topic specialization: ' + str(topics_specs2))
        print('level 3 topic specialization: ' + str(topics_specs3))
        beta = sess.run(model.beta)
        beta2 = sess.run(model.beta2)
        hierarchical_affinity3 = utils.compute_hierarchical_affinity(beta2, beta, sess.run((model.depend),{model.decay:0.1}),[effective_dims_bool2, effective_dims_bool])
        print('hierarchical_affinity3: ' + str(hierarchical_affinity3))
        CL_set = []
        OL_set = []
        calculate_CLNPMI(tree, data_mat, CL_set, OL_set)
        print('CLNPMI = ', str(np.mean(CL_set)))
        print('overlap = ', str(np.mean(OL_set)))

class Node(object):
    def __init__(self, beta = None,  depth = 0):
        self.beta = beta
        self.childs = []
        self.depth = depth

def build_tree(sess, model, effective_list):
    root = Node()
    par = np.argmax(sess.run(model.depend2),1)
    par2 = np.argmax(sess.run((model.depend),{model.decay:0.1}),1)
    Beta1 = sess.run(model.beta)
    Beta2 = sess.run(model.beta2)
    Beta3 = sess.run(model.beta3)

    for i,beta3 in enumerate(Beta3):
        childs = par==i
        level1 = Node(beta=beta3, depth=1)
        cnt = 0
        for j, flag in enumerate(childs):
            if flag == 1:
                level2 = Node(beta=Beta2[j], depth=2)
                childs2 = (par2 == j)*effective_list[0]
                if np.sum(childs2)>0:
                    for k, flag2 in enumerate(childs2):
                        if flag2 == 1:
                            level3 = Node(beta=Beta1[k], depth=3)
                            level2.childs.append(level3)
                    level1.childs.append(level2)
                    cnt += 1
        if cnt >0:
            root.childs.append(level1)
    return root

def print_tree(Node,vocab):
    if Node.depth != 0:

        phi = Node.beta.tolist()
        words = map(phi.index, heapq.nlargest(10, phi))
        words10 = []
        s = '   '*Node.depth+'level ' +  str(Node.depth)
        for w in words:
            words10.append(vocab[w])
            s += ' '+vocab[w]
        print(s)
    for child in Node.childs:
        print_tree(child,vocab)

def get_topics(Node, beta_list):
    if Node.depth != 0:
        beta_list[Node.depth-1].append(Node.beta.tolist())
    for child in Node.childs:
        get_topics(child, beta_list)

def calculate_CLNPMI(par,data_mat, CL_set, OL_set):
    for child in par.childs:
        if par.depth>0:
            clnpmi = utils.cal_clnpmi(par.beta, child.beta, data_mat)
            overlap = utils.cal_overlap(par.beta, child.beta)
            CL_set.append(clnpmi)
            OL_set.append(overlap)
        calculate_CLNPMI(child, data_mat, CL_set,OL_set)

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if FLAGS.non_linearity == 'tanh':
      non_linearity = tf.nn.tanh
    elif FLAGS.non_linearity == 'sigmoid':
      non_linearity = tf.nn.sigmoid
    else:
      non_linearity = tf.nn.relu

    ntsntm = nTSNTM(prior_alpha=FLAGS.prior_alpha,
                prior_beta=FLAGS.prior_beta,
                vocab_size=FLAGS.vocab_size,
                n_hidden=FLAGS.n_hidden,
                truncation_level=FLAGS.truncation_level,
                n_topic2=FLAGS.n_topic2,
                learning_rate=FLAGS.learning_rate, 
                batch_size=FLAGS.batch_size,
                non_linearity=non_linearity)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.initialize_all_variables()
    sess.run(init)
    train_url = os.path.join(FLAGS.data_dir, 'train.feat')
    test_url = os.path.join(FLAGS.data_dir, 'test.feat')
    train(sess, ntsntm, train_url, test_url, FLAGS.batch_size)

if __name__ == '__main__':
    tf.app.run()

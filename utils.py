import tensorflow as tf
import random
import numpy as np
import utils

def data_set(data_url, vocab_size):
    """process data input."""
    data_list = []
    word_count = []
    with open(data_url) as fin:
      while True:
        line = fin.readline()
        if not line:
          break
        id_freqs = line.split()
        id_freqs = id_freqs[0:-1]
        doc = {}
        count = 0
        for id_freq in id_freqs:
          items = id_freq.split(':')
          doc[int(items[0]) - 1] = int(items[1])
          count += int(items[1])
        if count > 0:
          data_list.append(doc)
          word_count.append(count)
    data_mat = np.zeros((len(data_list), vocab_size), dtype=np.int16)
    for doc_idx, doc in enumerate(data_list):
      for word_idx, count in doc.items():
        data_mat[doc_idx, word_idx] += count
    return data_list, data_mat, word_count

def create_batches(data_size, batch_size, shuffle=True):
  batches = []
  ids = list(range(data_size))
  if shuffle:
    random.shuffle(ids)
  for i in range(int(data_size / batch_size)):
    start = i * batch_size
    end = (i + 1) * batch_size
    batches.append(ids[start:end])
  # the batch of which the length is less than batch_size
  rest = data_size % batch_size
  if rest > 0:
    batches.append(ids[-rest:] + [-1] * (batch_size - rest))  # -1 as padding
  return batches

def fetch_data(data, count, idx_batch, vocab_size):
  """fetch input data by batch."""
  batch_size = len(idx_batch)
  data_batch = np.zeros((batch_size, vocab_size))
  count_batch = []
  mask = np.zeros(batch_size)
  for i, doc_id in enumerate(idx_batch):
    if doc_id != -1:
      for word_id, freq in data[doc_id].items():
        data_batch[i, word_id] = freq
      count_batch.append(count[doc_id])
      mask[i]=1.0
    else:
      count_batch.append(0)
  return data_batch, count_batch, mask

def variable_parser(var_list, prefix):
  """return a subset of the all_variables by prefix."""
  ret_list = []
  for var in var_list:
    varname = var.name
    varprefix = varname.split('/')[0]
    if varprefix == prefix:
      ret_list.append(var)
  return ret_list

def linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None):
  """define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer = None
    if bias_start_zero:
      bias_initializer = tf.constant_initializer(0)
    else:
      bias_initializer = None
    input_size = inputs.get_shape()[1].value
    matrix = tf.get_variable('Matrix', [input_size, output_size],
                             initializer=matrix_initializer)
    bias_term = tf.get_variable('Bias', [output_size], 
                                initializer=bias_initializer)
    output = tf.matmul(inputs, matrix)
    if not no_bias:
      output = output + bias_term
  return output

def small_linear(inputs,
           output_size,
           no_bias=False,
           bias_start_zero=False,
           matrix_start_zero=False,
           scope=None):
  """define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    input_size = inputs.get_shape()[1].value
    matrix = tf.get_variable('Matrix', [input_size, output_size],
                             initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=0))
    bias_term = tf.get_variable('Bias', [output_size],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.01, seed=0))
    output = tf.matmul(inputs, matrix)
    if not no_bias:
      output = output + bias_term
  return output

def mlp(inputs, 
        mlp_hidden=[], 
        mlp_nonlinearity=tf.nn.tanh,
        scope=None):
  """define an MLP."""
  with tf.variable_scope(scope or 'Linear'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l)))
    return res

def conv(inputs,
           output_size,
           mask,
           matrix_start_zero=False,
           scope=None):
  """define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    if matrix_start_zero:
      matrix_initializer = tf.constant_initializer(0)
    else:
      matrix_initializer = None
    input_size = inputs.get_shape()[1].value
    matrix = tf.get_variable('Matrix', [input_size, output_size],
                             initializer=matrix_initializer)
    matrix = matrix*mask
    output = tf.matmul(inputs, matrix)
  return output

def compute_TU(topic_word, N):
  topic_size, word_size = np.shape(topic_word)
  topic_list = []
  for topic_idx in range(topic_size):
    top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
    topic_list.append(top_word_idx)
  TU= 0
  cnt =[0 for i in range(word_size)]
  for topic in topic_list:
    for word in topic:
      cnt[word]+=1
  for topic in topic_list:
    TU_t = 0
    for word in topic:
      TU_t+=1/cnt[word]
    TU_t/=N
    TU+=TU_t
  TU/=topic_size
  return TU

def evaluate_coherence(topic_word, doc_word, N_list):
    topic_size = len(topic_word)
    doc_size = len(doc_word)

    average_coherence = 0.0
    for N in N_list:
      # find top words' index of each topic
      topic_list = []
      for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)

      # compute coherence of each topic
      sum_coherence_score = 0.0
      for i in range(topic_size):
        word_array = topic_list[i]
        sum_score = 0.0
        for n in range(N):
          flag_n = doc_word[:, word_array[n]] > 0
          p_n = np.sum(flag_n) / doc_size
          for l in range(n + 1, N):
            flag_l = doc_word[:, word_array[l]] > 0
            p_l = np.sum(flag_l)
            p_nl = np.sum(flag_n * flag_l)
            if p_nl == doc_size:
              sum_score += 1
            elif p_n > 0 and p_l>0 and p_nl>0:
              p_l = p_l / doc_size
              p_nl = p_nl / doc_size
              sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        sum_coherence_score += sum_score * (2 / (N * N - N))
      sum_coherence_score = sum_coherence_score / topic_size
      average_coherence += sum_coherence_score
    average_coherence /= len(N_list)
    return average_coherence

def evaluate_TU(topic_word,  n_list):
    TU = 0.0
    for n in n_list:
        TU += compute_TU(topic_word, n)
    TU /= len(n_list)
    return TU

def compute_topic_specialization(topic_word, corpus_topic):
  topics_vec = topic_word
  for i in range(topics_vec.shape[0]):
    topics_vec[i] = topics_vec[i]/np.linalg.norm(topics_vec[i])
  topics_spec = 1 - topics_vec.dot(corpus_topic)
  depth_spec = np.mean(topics_spec)
  return depth_spec

def compute_hierarchical_affinity(parent_topic_word, child_topic_word, dependency_matrix,effective_bool_list):
  for i in range(parent_topic_word.shape[0]):
    parent_topic_word[i] = parent_topic_word[i]/np.linalg.norm(parent_topic_word[i])
  for i in range(child_topic_word.shape[0]):
    child_topic_word[i] = child_topic_word[i]/np.linalg.norm(child_topic_word[i])

  child_ha = []
  unchild_ha = []
  parent_index =  np.argmax(dependency_matrix, axis=1)
  for i in range(parent_topic_word.shape[0]):
    if effective_bool_list[0][i]:
      child_index = (parent_index == i)*effective_bool_list[1]
      for j in range(len(child_index)):
        if child_index[j]:
          child_ha.append(parent_topic_word[i].dot(child_topic_word[j]))
        elif effective_bool_list[1][j]:
          unchild_ha.append(parent_topic_word[i].dot(child_topic_word[j]))
  return np.mean(child_ha), np.mean(unchild_ha)

def get_vocab(url):
  vocab = {}
  with open(url, 'r') as file_to_read:
    i = 0
    while True:
      lines = file_to_read.readline()
      if not lines:
        break
      word, num = lines.split()
      vocab[word] = i
      i += 1
  return vocab, i

def cal_clnpmi(level1, level2, all_set):

  sum_coherence_score = 0.0
  c = 0
  for N in [5, 10, 15]:
    word_idx1 = np.argpartition(level1, -N)[-N:]
    word_idx2 = np.argpartition(level2, -N)[-N:]
    sum_score = 0.0
    set1 = set(word_idx1)
    set2 = set(word_idx2)
    inter = set1.intersection(set2)
    word_idx1 = list(set1.difference(inter))
    word_idx2 = list(set2.difference(inter))

    for n in range(len(word_idx1)):
      flag_n = all_set[:, word_idx1[n]] > 0
      p_n = np.sum(flag_n) / len(all_set)
      for l in range(len(word_idx2)):
        flag_l = all_set[:, word_idx2[l]] > 0
        p_l = np.sum(flag_l)
        p_nl = np.sum(flag_n * flag_l)
        if p_nl == len(all_set):
          sum_score += 1
        else:
          p_l = p_l / len(all_set)
          p_nl = p_nl / len(all_set)
          p_nl += 1e-10
          sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        c += 1
    if c>0:
      sum_score /= c
    else:
      sum_score = 0
    sum_coherence_score += sum_score
  return sum_coherence_score / 3

def cal_overlap(level1, level2):
  sum_overlap_score = 0.0
  for N in [5, 10, 15]:
    word_idx1 = np.argpartition(level1, -N)[-N:]
    word_idx2 = np.argpartition(level2, -N)[-N:]
    c = 0
    for n in word_idx1:
      if n in word_idx2:
        c+=1
    sum_overlap_score += c/N
  return sum_overlap_score/3

def xavier_init(fan_in, fan_out, constant=1):
  low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
  high = constant * np.sqrt(6.0 / (fan_in + fan_out))
  return tf.random_uniform((fan_in, fan_out),
                           minval=low, maxval=high,
                           dtype=tf.float32)

def softmax(x):
  x_exp = np.exp(x)
  x_sum = np.sum(x_exp, axis=1, keepdims=True)
  s = x_exp / x_sum
  return s
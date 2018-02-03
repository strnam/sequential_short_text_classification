import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import os
from sklearn.metrics import f1_score


def batch_iter(data, batch_size, num_epochs, shuffle=True):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    if shuffle:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffled_data = data[shuffle_indices]
    else:
      shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffled_data[start_index:end_index]


def compute_dialogue_act_acc(labels, labels_pred, dialogue_lengths):
  correct_preds, total_correct, total_preds = 0., 0., 0.
  accs = []
  for lab, lab_pred, length in zip(labels, labels_pred,
                                   dialogue_lengths):
    lab = lab[:length]
    lab_pred = lab_pred[:length]
    accs += [ a==b for (a, b) in zip(lab, lab_pred)]
  acc = np.mean(accs)
  return acc


def tf_shift_utts_in_session(sessions, num_shift_row=2, max_utterance_in_session = 23):
  """
  Input
  [[1, 1, 1, 1],
  [1, 2, 2, 2],
  [1, 3, 3, 3],
  [1, 4, 4, 4],
  [1,5,5,5]]

  Output
  [[0 0 0 0]
  [0 0 0 0]
  [1 1 1 1]
  [1 2 2 2]
  [1 3 3 3]
  """
  diag_vec = [1]*(max_utterance_in_session - num_shift_row) + [0]*num_shift_row
  diag = tf.diag(diag_vec)
  diag = tf.cast(diag, dtype=tf.float32)
  zero_tail_session = tf.transpose(tf.matmul(sessions, diag, transpose_a=True))
  zero_indices = [max_utterance_in_session - i - 1 for i in range(num_shift_row)]
  new_indices = tf.constant(zero_indices + list(range(0, max_utterance_in_session - num_shift_row)), dtype=tf.int32)
  zero_head_session = tf.gather(zero_tail_session, new_indices)
  return zero_head_session


class SeqShortextClassifcation(object):
  def __init__(self):
    self.sess = None
    self.saver = None

    self.num_hiden_units = 100
    self.max_utterance_in_session = 536
    self.max_word_in_utterance = 200
    self.num_classes = 43
    self.embedding_size = 100
    self.vocab_size = 10000
    self.utterance_filter_sizes = [3, 4, 5]
    self.utterance_num_filter = 8
    self.session_filter_sizes = [2, 3]
    self.session_num_filters = 8
    self.use_crf = False
    self.num_checkpoint = 5
    self.evaluate_every = 100
    self.batch_size = 8
    self.num_epochs = 40
    self.checkpoint_every = 100
    self.build_model()


  def build_model(self):
    self.add_input()
    self.add_embedding()
    self.add_utterance_model()
    self.add_disclosure_model()
    self.add_loss_op()
    self.add_accuracy_op()
    self.add_train_op()
    self.initialize_session()

  def add_train_op(self):
    with tf.variable_scope("train_step"):
      optimizer = tf.train.AdamOptimizer(1e-3)
      self.global_step = tf.Variable(0, name="global_step", trainable=False)
      self.grads_and_vars = optimizer.compute_gradients(self.loss)
      self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)


  def add_input(self):
    # Init Variable
    self.input_x = tf.placeholder(tf.int32, shape=[None, self.max_utterance_in_session, self.max_word_in_utterance], name="input_x_raw")
    self.input_y = tf.placeholder(tf.float32, [None, self.max_utterance_in_session, self.num_classes], name="input_y")
    self.session_lengths = tf.placeholder(tf.int32, [None], name="session_lenght")
    self.word_ids = tf.reshape(self.input_x, [-1, self.max_word_in_utterance])


  def add_embedding(self):
    # Word Embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
      self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], minval=-1, maxval=1), name='W')
      # [batch*max_utterance_in_session, max_word_in_utterance, embedding_size]
      self.word_embedding = tf.nn.embedding_lookup(self.W, self.word_ids)
      self.word_embedding = tf.expand_dims(self.word_embedding, -1)
      print("embedding re-dim: ", self.word_embedding.get_shape())


  def add_utterance_model(self):
    # Utterance representation
    applied_conv_on_sentence_outputs = []
    for i, filter_size in enumerate(self.utterance_filter_sizes):
      with tf.name_scope('utterance-conv-maxpool-%s' % filter_size):
        filter_shape = [filter_size, self.embedding_size, 1, self.utterance_num_filter]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[self.utterance_num_filter]), name="b")
        conv = tf.nn.conv2d(
          self.word_embedding,
          W,
          strides=[1, 1, 1, 1],
          padding="VALID",
          name="conv")
        # Apply nonlinearity
        print(conv.get_shape())
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #Maxpooling over the outputs.
        pooled = tf.nn.max_pool(
          h,
          ksize=[1, self.max_word_in_utterance - filter_size + 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding="VALID"
        )
        # pooled has shape: [batch*max_utterance_in_session, 1, 1, utterance_num_filter]
        applied_conv_on_sentence_outputs.append(pooled)

    self.num_utterance_filters_total = self.utterance_num_filter*len(self.utterance_filter_sizes)
    utterance_vector = tf.concat(applied_conv_on_sentence_outputs, 3)
    # shape [batch * max_utterance_in_session, num_utterance_filters_total]
    #self.utterance_vector = tf.reshape(utterance_vector, [-1, num_utterance_filters_total])
    self.utterance_vector_group_by_sess = tf.reshape(utterance_vector, [-1, self.max_utterance_in_session, self.num_utterance_filters_total])

  def add_disclosure_model(self):
    utterance_vector_length = self.num_utterance_filters_total
    with tf.name_scope("Classification"):
      # Layer 1
      layer1_W_minus_2 = tf.get_variable("layer1_W_2", shape=[utterance_vector_length, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      layer1_W_minus_1 = tf.get_variable("layer1_W_1", shape=[utterance_vector_length, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      layer1_W = tf.get_variable("layer1_W", shape=[utterance_vector_length, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      layer1_b = tf.get_variable("layer1_b", shape=[self.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())


      # Layer 2
      layer2_W_minus_2 = tf.get_variable("layer2_W_2", shape=[self.num_classes, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      layer2_W_minus_1 = tf.get_variable("layer2_W_1", shape=[self.num_classes, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      layer2_W = tf.get_variable("layer2_W", shape=[self.num_classes, self.num_classes], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
      layer2_b = tf.get_variable("layer2_b", shape=[self.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer())

      num_sess = self.input_x.shape[0]
      sess_mat_minus2 = tf.map_fn(lambda chat_sess: tf_shift_utts_in_session(chat_sess, num_shift_row=2, max_utterance_in_session=self.max_utterance_in_session),
                                  self.utterance_vector_group_by_sess)
      sess_mat_minus1 = tf.map_fn(lambda chat_sess: tf_shift_utts_in_session(chat_sess, num_shift_row=1, max_utterance_in_session=self.max_utterance_in_session),
                                  self.utterance_vector_group_by_sess)

      sess_mat_minus2_flat = tf.reshape(sess_mat_minus2, [-1, utterance_vector_length])
      sess_mat_minus1_flat = tf.reshape(sess_mat_minus1, [-1, utterance_vector_length])
      sess_mat_flat = tf.reshape(self.utterance_vector_group_by_sess, [-1, utterance_vector_length])


      layer1_mat_minus_2 = tf.matmul(sess_mat_minus2_flat, layer1_W_minus_2)
      layer1_mat_minus_1 = tf.matmul(sess_mat_minus1_flat, layer1_W_minus_1)
      layer1_mat_mul = tf.matmul(sess_mat_flat, layer1_W)
      y_layer1 = tf.tanh(layer1_mat_minus_2 + layer1_mat_minus_1 + layer1_mat_mul + layer1_b)

      # layer 2
      y_layer1_reshape = tf.reshape(layer1_mat_minus_2, [-1, self.max_utterance_in_session, self.num_classes])

      layer2_sess_mat_minus2 = tf.map_fn(lambda chat_sess: tf_shift_utts_in_session(chat_sess, num_shift_row=2, max_utterance_in_session=self.max_utterance_in_session),
                                         y_layer1_reshape)
      layer2_sess_mat_minus1 = tf.map_fn(lambda chat_sess: tf_shift_utts_in_session(chat_sess, num_shift_row=1, max_utterance_in_session=self.max_utterance_in_session),
                                         y_layer1_reshape)

      layer2_sess_mat_minus2_flat = tf.reshape(layer2_sess_mat_minus2, [-1, self.num_classes])
      layer2_sess_mat_minus1_flat = tf.reshape(layer2_sess_mat_minus1, [-1, self.num_classes])
      layer2_sess_mat_flat = y_layer1

      layer2_mat_minus_2 = tf.matmul(layer2_sess_mat_minus2_flat, layer2_W_minus_2)
      layer2_mat_minus_1 = tf.matmul(layer2_sess_mat_minus1_flat, layer2_W_minus_1)
      layer2_mat_mul = tf.matmul(layer2_sess_mat_flat, layer2_W)
      pred = (layer2_mat_minus_2 + layer2_mat_minus_1 + layer2_mat_mul + layer2_b)
      self.scores = tf.reshape(pred, [-1, self.max_utterance_in_session, self.num_classes])
      self.mask = tf.sequence_mask(self.session_lengths, maxlen=self.max_utterance_in_session)


  def add_loss_op(self):
    with tf.name_scope("loss"):
      if self.use_crf:
        labels = tf.argmax(self.input_y, axis=2)
        labels = tf.cast(labels, dtype=tf.int32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
          self.scores, labels, self.session_lengths)
        self.transition_params = transition_params
        self.loss = tf.reduce_mean(-log_likelihood)
      else:
        y_by_utterance = tf.boolean_mask(self.input_y, self.mask)
        scores = tf.boolean_mask(self.scores, self.mask)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y_by_utterance)
        self.loss = tf.reduce_mean(losses)


  def add_accuracy_op(self):
    if not self.use_crf:
      with tf.name_scope("Accuracy"):
        self.predictions = tf.boolean_mask(self.scores, self.mask)
        self.predictions = tf.argmax(self.predictions, 1, name="predictions")
        y_by_utterance = tf.boolean_mask(self.input_y, self.mask)
        self.y_arg_max = tf.argmax(y_by_utterance, 1)
        print('yarg_max', self.y_arg_max.get_shape())
        correct_predictions = tf.equal(self.predictions, self.y_arg_max)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

  def initialize_session(self):
    print("Initializing tf session")
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.sess.run(tf.local_variables_initializer())
    self.saver = tf.train.Saver(max_to_keep=self.num_checkpoint)

  def add_summary(self):
    # Difine dir
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in self.grads_and_vars:
      if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", self.loss)

    # Dev summaries
    if self.use_crf:
      self.train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
      self.dev_summary_op = tf.summary.merge([loss_summary])
    else:
      acc_summary = tf.summary.scalar("accuracy", self.accuracy)
      self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
      self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])

    # Train Summaries
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

  def train_step(self, x_batch, y_batch, x_sequence_lenght_batch):
    """
    A single training step
    """
    feed_dict = {
      self.input_x: x_batch,
      self.input_y: y_batch,
      self.session_lengths: x_sequence_lenght_batch,
    }

    if self.use_crf:
      _, step, summaries, loss = self.sess.run(
        [self.train_op, self.global_step, self.train_summary_op, self.loss],
        feed_dict)
      time_str = datetime.now().isoformat()
      print("{}: step {}, loss {:g}".format(time_str, step, loss))
      self.train_summary_writer.add_summary(summaries, step)
    else:
      _, step, summaries, loss, accuracy = self.sess.run(
        [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy],
        feed_dict)
      time_str = datetime.now().isoformat()
      print("New {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      self.train_summary_writer.add_summary(summaries, step)

  def dev_step(self, x_batch, y_batch, x_sequence_lenght_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      self.input_x: x_batch,
      self.input_y: y_batch,
      self.session_lengths: x_sequence_lenght_batch,
    }
    if self.use_crf:
      viterbi_sequences = []
      step, summaries, loss, logits, trans_params = self.sess.run([self.global_step, self.dev_summary_op, self.loss,
                                                                   self.scores, self.transition_params],
                                                                  feed_dict=feed_dict)
      for logit, sequence_length in zip(logits, x_sequence_lenght_batch):
        logit = logit[:sequence_length]  # keep only the valid steps
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
          logit, trans_params)
        viterbi_sequences += [viterbi_seq]

      time_str = datetime.now().isoformat()
      y_batch_label = np.argmax(y_batch, axis=2)
      acc = compute_dialogue_act_acc(y_batch_label, viterbi_sequences, x_sequence_lenght_batch)
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
    else:
      step, summaries, loss, accuracy = self.sess.run(
        [self.global_step, self.dev_summary_op, self.loss, self.accuracy],
        feed_dict)
      time_str = datetime.now().isoformat()
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

    if writer:
      writer.add_summary(summaries, step)

  def predict(self, x_batch, x_sequence_lenght_batch):
    feed_dict = {
      self.input_x: x_batch,
      self.session_lengths: x_sequence_lenght_batch,
    }
    pred, score = self.sess.run(
      [self.pred, self.scores_sigmoid_mask ],
      feed_dict)
    time_str = datetime.now().isoformat()
    return pred, score

  def train(self, train, dev=None):
    x_train, y_train, x_seq_len_train = train
    if dev:
      x_dev, y_dev, x_seq_len_dev = dev
    batches = batch_iter(list(zip(x_train, y_train, x_seq_len_train)), batch_size=self.batch_size, num_epochs=self.num_epochs)

    self.add_summary()  # tensorboard
    # Training loop. For each batch...
    for batch in batches:
      x_batch, y_batch, x_sequence_lenght_batch = zip(*batch)
      self.train_step(x_batch, y_batch, x_sequence_lenght_batch)
      current_step = tf.train.global_step(self.sess, self.global_step)
      if dev:
        if current_step % self.evaluate_every == 0:
          print("\nEvaluation:")
          self.dev_step(x_dev, y_dev, x_seq_len_dev, writer=self.dev_summary_writer)
          print("")

      if current_step % self.checkpoint_every == 0:
        path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))





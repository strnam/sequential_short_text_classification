from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer

from data_provider import dialogue_dataset
from models.seq_shortext_classification import SeqShortextClassifcation


DATASET_NAME = 'Switchboard'
num_vocabs = 10000
max_utterance_in_session = 536
max_word_in_utterance = 200



class Config(object):
  def __init__(self, dataset_name, load=True):
    self.dataset_name = dataset_name
    if load:
      self.load_data()

  def load_data(self):
    # LOAD DATA
    dataset = dialogue_dataset.get_dataset(self.dataset_name)

    # SPLIT DATA TO TRAIN, VALID, TEST
    self.train_ids, self.valid_ids, self.test_ids = dataset.split_data_to_train()


def main():
  # LOAD DATA
  print('Load data')
  config = Config(DATASET_NAME)
  dataset = dialogue_dataset.get_dataset(DATASET_NAME)

  x_train, y_train = dataset.get_dialogue_data(config.train_ids) # list[list], list[list]
  x_test, y_test = dataset.get_dialogue_data(config.test_ids)
  x_dev, y_dev = dataset.get_dialogue_data(config.valid_ids)

  # Store dialouge lenght
  x_train_dialogue_len = dataset.get_dialogues_length(x_train)
  x_test_dialogue_len = dataset.get_dialogues_length(x_test)
  x_dev_dialogue_len = dataset.get_dialogues_length(x_dev)

  x_train_flat = dataset.flat_dialogue(x_train)
  x_dev_flat = dataset.flat_dialogue(x_dev)

  tokenizer = Tokenizer(num_words=num_vocabs)
  tokenizer.fit_on_texts(x_train_flat)
  X_train_flat = tokenizer.texts_to_sequences(x_train_flat)
  X_dev_flat = tokenizer.texts_to_sequences(x_dev_flat)

  X_train = dataset.group_utterance_to_dialogue(X_train_flat, x_train_dialogue_len)
  X_dev = dataset.group_utterance_to_dialogue(X_dev_flat, x_dev_dialogue_len)

  y_train_one_hot = [to_categorical(y_session, num_classes=43) for y_session in y_train]
  y_dev_one_hot = [to_categorical(y_session, num_classes=43) for y_session in y_dev]

  X_train = [pad_sequences(session, padding='post', truncating='post', maxlen=max_word_in_utterance) for session in X_train]
  X_dev = [pad_sequences(session, padding='post', truncating='post',  maxlen=max_word_in_utterance) for session in X_dev]


  ## padding utterance in session
  X_train = pad_sequences(X_train,  padding='post', truncating='post', maxlen=max_utterance_in_session)
  X_dev = pad_sequences(X_dev, padding='post', truncating='post', maxlen=max_utterance_in_session)
  y_train = pad_sequences(y_train_one_hot, padding='post', truncating='post', maxlen=max_utterance_in_session)
  y_dev = pad_sequences(y_dev_one_hot, padding='post', truncating='post', maxlen=max_utterance_in_session)


  model = SeqShortextClassifcation()
  train = (X_train, y_train, x_train_dialogue_len)
  dev = (X_dev, y_dev, x_dev_dialogue_len)
  model.train(train, dev)


if __name__ == '__main__':
  main()



import pandas as pd
import numpy as np
from pathlib import Path, PureWindowsPath
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras import layers
from keras import utils


def sample_file_percentage(pathin, pathout, percentage):
    """ Description
    Returns a random sample of length file path * percentage % without replacement by opening the file in the path
    """
    df = pd.read_csv(pathin, names=["id", "text", "label"], sep=",");
    
    elements_n = int(len(df.values) * percentage / 100)

    dff =  pd.DataFrame(df.values)
    sampled = dff.sample(n=elements_n, replace=False)
    print(len(sampled))
    #write to file
    sampled.to_csv(pathout, index=False, index_label=False)
#file_path = PureWindowsPath("E:\Users\User\Documents\SCHOOL\\5thYear\data_wrangling\cw-partB\short.csv")
path_full = "E:\\Users\\User\\Documents\\SCHOOL\\5thYear\\data_wrangling\\cw-partB\\train.csv"
path_sampled = "E:\\Users\\User\\Documents\\SCHOOL\\5thYear\\data_wrangling\\cw-partB\\sampled.csv"

file_path = path_sampled



# if file_path.exists():
#     print("ok")
# else:
#     raise FileNotFoundError("the file specified doesn't exist!")

#hyperparameters
max_len = 50
embedding_length = 50 #describe how many elements a word vector will have
filter_n = 25
filter_heigth = 10
strides = 1
batch_size = 32
epochs = 10
#


df = pd.read_csv(file_path, names=["id", "text", "label"], sep=",");
num_class = len(np.unique(df["label"].values))
print(df.iloc[0])

data = df["text"].values
labels = df["label"].values

labels_int = LabelEncoder().fit(labels).transform(labels)
#1hot encode the labels
labels_encoded = utils.to_categorical(labels_int)
data_train, data_test, label_train, label_test = train_test_split(data, labels_encoded, test_size=0.20)

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data_train);
x_train = tokenizer.texts_to_sequences(data_train);
x_test = tokenizer.texts_to_sequences(data_test);

vocab_size = len(tokenizer.word_index) + 1
print(data_train[0])
print(x_train[0])

data_train_seq = sequence.pad_sequences(x_train, maxlen=max_len) #no maxlen at the moment
data_test_seq = sequence.pad_sequences(x_test, maxlen=max_len)

print(data_train_seq[0])
print(data_train_seq.shape[1])
print(len(data_train_seq[0]))
print(len(data_train_seq))

model = Sequential()
model.add(layers.Embedding(vocab_size,embedding_length , input_length=data_train_seq.shape[1], trainable=True))
model.add(layers.Conv1D(filter_n, filter_heigth, strides=strides, padding='valid', activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dropout(0.5))
#model.add(layers.Dense(10, activation='relu'))#sigmoid for multicalss, softmax for single classes
model.add(layers.Dense(3, activation='softmax'))#sigmoid for multicalss, softmax for single classes
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

training = model.fit(data_train_seq, label_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test_seq, label_test))

loss, accuracy = model.evaluate(data_train_seq, label_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(data_test_seq, label_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))
# print(data_test.shape)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(training):
    acc = training.history['accuracy']
    val_acc = training.history['val_accuracy']
    loss = training.history['loss']
    val_loss = training.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 2)
    plt.plot(x, acc, 'b', label='Training acc', color="blue")
    plt.plot(x, val_acc, 'r', label='Validation acc', color="green")
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss', color="pink")
    plt.plot(x, val_loss, 'r', label='Validation loss', color="red")
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


training.history
plot_history(training)





def clean_dict():
    pass

def word2vec_embedding():
    pass

def word2vec_pretrained_embedding():
    pass
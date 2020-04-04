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
#from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def sample_file_percentage(pathin, directory, percentage):
    """ Description:
    Returns a random sample of length file path * percentage % without replacement by opening the file in the path
    """
    df = pd.read_csv(pathin, names=["id", "text", "label"], sep=",");
    
    elements_n = int(len(df.values) * percentage / 100)

    dff =  pd.DataFrame(df.values)
    sampled = dff.sample(n=elements_n, replace=False)
    print(len(sampled))
    #write to file - still have to clean manually the columns names!
    sampled.to_csv(directory + "sampled" + str(percentage) + ".csv", index=False, index_label=False)


def set_up_model(vocab_size, input_length):
    model = Sequential()
    model.add(layers.Embedding(vocab_size,embedding_length , input_length=input_length, trainable=False))
    model.add(layers.Conv1D(filter_n, filter_heigth, strides=strides, padding='valid', activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(10, activation='relu'))#sigmoid for multicalss, softmax for single classes
    model.add(layers.Dense(3, activation='softmax'))#sigmoid for multicalss, softmax for single classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def print_info_labels(labels, axis=None):
    classes, classes_c = np.unique(labels, return_counts=True, axis=axis)
    print("number of labels: " + str(len(classes)))
    print("labels available and occurrences: ")
    for i, v in enumerate(classes):
        print(v, end='')
        print(" : ", end=' ')
        print(classes_c[i])
    return classes_c, classes 

def train_model():
    pass

def calculate_confusion_matrix(correct_labels, prediction_labels, correct1hot=True, prediction1hot=False, names=None):
    """Description: print to screen confusion matrix of predicted labels compared to correct ones. Correct labels can be 1hotencoded """
    if(correct1hot):
        correct_labels = np.argmax(correct_labels, axis=1)
    if(prediction1hot):
        prediction_labels = np.argmax(prediction_labels, axis=1)
            #Predicted0|Predicted1|Predict3|
    #Actual0|          |          |        |
    #Actual1|          |          |        |
    #Actual2|          |          |        |
    c_matrix = confusion_matrix(correct_labels, prediction_labels, labels=names)
    print(c_matrix)
    print("Accuracy score: ", accuracy_score(correct_labels, prediction_labels))
    print("Report")
    print(classification_report(correct_labels, prediction_labels))


def word2vec_embedding():
    pass

def run_kfold(splits, data, labels):
    
    kf = KFold(n_splits=splits, shuffle=True)
    encoder = LabelEncoder()
    labels_int = encoder.fit(labels).transform(labels)
    print(encoder.classes_)
    label_names = encoder.classes_
    #1hot encode the labels - [1,0,0] : 0; [0,1,0]: 1; [0,0,1]: 2
    labels_encoded = utils.to_categorical(labels_int)
    history = list()
    for train_index, test_index in kf.split(data):
        #data is splitted based on the 
        data_train , data_test = data[train_index], data[test_index]
        label_train , label_test = labels_encoded[train_index], labels_encoded[test_index]
        
        #resetting the model every iteration
        history.append(run_model(data_train, data_test, label_train, label_test))
    return history


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



#file_path = PureWindowsPath("E:\Users\User\Documents\SCHOOL\\5thYear\data_wrangling\cw-partB\short.csv")
path_full = "E:\\Users\\User\\Documents\\SCHOOL\\5thYear\\data_wrangling\\cw-partB\\train.csv"
directory = "E:\\Users\\User\\Documents\\SCHOOL\\5thYear\\data_wrangling\\cw-partB\\"
path_sampled = "E:\\Users\\User\\Documents\\SCHOOL\\5thYear\\data_wrangling\\cw-partB\\sampled.csv"
path_sampled5 = "E:\\Users\\User\\Documents\\SCHOOL\\5thYear\\data_wrangling\\cw-partB\\sampled5.csv"


file_path = path_sampled5



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

def run_model(data_train, data_test, label_train, label_test):
    print("Training labels")
    print_info_labels(label_train,0)
    print("Testing labels")
    print_info_labels(label_test,0)
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(data_train)
    x_train = tokenizer.texts_to_sequences(data_train)
    x_test = tokenizer.texts_to_sequences(data_test)

    vocab_size = len(tokenizer.word_index) + 1
    print(data_train[0])
    print(x_train[0])

    data_train_seq = sequence.pad_sequences(x_train, maxlen=max_len) #no maxlen at the moment
    data_test_seq = sequence.pad_sequences(x_test, maxlen=max_len)

    print(data_train_seq[0])
    print(data_train_seq.shape[1])
    print(len(data_train_seq[0]))
    print(len(data_train_seq))

    model = set_up_model(vocab_size, data_train_seq.shape[1])
    training = model.fit(data_train_seq, label_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test_seq, label_test))

    loss, accuracy = model.evaluate(data_train_seq, label_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(data_test_seq, label_test, verbose=True)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    # print(data_test.shape)
    #debug trying to predict specific ones
    predictions_class = model.predict_classes(data_train_seq, batch_size)#, len(data_train_seq) // batch_size + 1 )

    print("predictions:", pd.unique(predictions_class))
    calculate_confusion_matrix(label_train, predictions_class)


    predictions = model.predict_classes(data_test_seq, batch_size)
    print("predictions:", pd.unique(predictions))
    calculate_confusion_matrix(label_test, predictions)
    return training

def run_single_model(data, labels):
    #model = set_up_model()
    encoder = LabelEncoder()
    labels_int = encoder.fit(labels).transform(labels)
    print(encoder.classes_)
    label_names = encoder.classes_
    #1hot encode the labels - [1,0,0] : 0; [0,1,0]: 1; [0,0,1]: 2
    labels_encoded = utils.to_categorical(labels_int)
    data_train, data_test, label_train, label_test = train_test_split(data, labels_encoded, test_size=0.20)
    return run_model(data_train, data_test, label_train, label_test)


def main():
    df = pd.read_csv(file_path, names=["id", "text", "label"], sep=",");

    data = df["text"].values
    labels = df["label"].values

    print("total labels")
    labels_n, __ = print_info_labels(labels)

    #training = run_single_model(data, labels)
    history = run_kfold(2, data, labels)
    

    plt.style.use('ggplot')
    for training in history:
        plot_history(training)










def clean_dict():
    pass



def word2vec_pretrained_embedding():
    pass

if __name__ == "__main__":
    main()
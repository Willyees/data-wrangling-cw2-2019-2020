import pandas as pd
from pandas import DataFrame
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
#word2vec
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from keras_preprocessing.text import text_to_word_sequence
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
#from spellchecker import SpellChecker

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


def set_up_model(vocab_size, input_length, embedding=np.zeros((1,0))):
    """embedding optional"""
    model = Sequential()
    if(embedding.size):
        model.add(layers.Embedding(vocab_size,embedding_length , input_length=input_length, weights=[embedding], trainable=False))
    else:
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




def run_kfold(splits, data, labels, word2vec=True, pre_embedding=False):
    
    kf = KFold(n_splits=splits, shuffle=True)
    encoder = LabelEncoder()
    labels_int = encoder.fit(labels).transform(labels)
    print(encoder.classes_)
    label_names = encoder.classes_
    #1hot encode the labels - [1,0,0] : 0; [0,1,0]: 1; [0,0,1]: 2
    labels_encoded = utils.to_categorical(labels_int)
    history = list()
    accuracies_training = list()
    accuracies_testing = list()

    for train_index, test_index in kf.split(data):
        #data is splitted based on the 
        data_train , data_test = data[train_index], data[test_index]
        label_train , label_test = labels_encoded[train_index], labels_encoded[test_index]
        
        #resetting the model every iteration
        history_temp, a_tr, a_te = run_model(data_train, data_test, label_train, label_test, word2vec, pre_embedding)
        accuracies_training.append(a_tr)
        accuracies_testing.append(a_te)
        history.append(history_temp)
    print("KFold ended")
    print("Accuracy training average: {}".format(str(average_accuracies(accuracies_training)[0])))
    print("Accuracy testing average: {}".format(str(average_accuracies(accuracies_testing)[0])))
    return history, accuracies_training, accuracies_testing

def average_accuracies(accuracies):
    """Assumes that accuracies is a list of tuples (accuracy, loss)"""
    average_accuracy = np.average(list(list(zip(*accuracies))[0]))
    average_loss = np.average(list(list(zip(*accuracies))[1]))
    return average_accuracy, average_loss

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






# if file_path.exists():
#     print("ok")
# else:
#     raise FileNotFoundError("the file specified doesn't exist!")

#hyperparameters
max_len = 100 #number of words per sentences. If sentence has less, then added wv empty <0,0,0>
embedding_length = 50 #describe how many elements a word vector will have
filter_n = 100
filter_heigth = 10
strides = 1
batch_size = 32
epochs = 10
#
#path_model_wc = 'E:\\Users\\User\\Documents\\SCHOOL\\5thYear\\data_wrangling\\cw-partB\\models\\init_50.wv' #model to be used to load a pretrained wc
#path_model_wc = 'C:\\Users\\Alessio\\gensim-data\\glove-twitter-25\\glove-twitter-25.gz'
path_model_wc = 'C:\\Users\\Alessio\\gensim-data\\glove-wiki-gigaword-50\\glove-wiki-gigaword-50.gz'

def word2vec_embedding(vocab_size, word_index, pre_embedding=False, data_embedding=np.zeros((1,0))):
    """data_embedding: optional, only needed if pre_embedding is not used. pre_embedding states that a preembedding generated model will be loaded from 
    global variable path_model_wc (bad practice fix). data_embedding used to pass the data to train the word embedding"""
    #load pretrained model
    print("Fitting the embedding matrix")
    word_vectors : KeyedVectors
    if not pre_embedding:
        print("Calculating the words vector from datasetfile")
        if not data_embedding.size:
            #list is empty and should not be: throw error
            raise AttributeError("data embedding parameter should not be empty if pre_embedding is set to False")
        word_vectors = get_word2vec_embedding(data_embedding)
    else:
        print("Loading preembedded word vector from {}".format(path_model_wc))
        word_vectors = KeyedVectors.load_word2vec_format(path_model_wc, binary=False)
    embedding_matrix = np.zeros((vocab_size, embedding_length))

    c_not_present = 0
    for word, position in word_index.items():
        #check that vocab_size is 2000
        if position >= vocab_size:
            continue
        try:
            embedding_word = word_vectors[word]
            embedding_matrix[position] = embedding_word
        except KeyError:
            #create random word vector - can check if there is difference by random and 0 intialized
            embedding_matrix[position] = np.random.normal(scale = 0.1,size = (embedding_length, ))
            c_not_present += 1
    print("Words not present, total intialized at random: {} %".format(str(c_not_present * 100 / len(embedding_matrix))))
    return embedding_matrix


def run_model(data_train, data_test, label_train, label_test, word2vec=True, pre_embedding=False):
    """Return: history: of training, evaluation_training: (accuracy, loss), evaluation_testing : (accuracy, loss). If embedding is set to true, prembedding words matrix is calculated """
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
    #set it to none. Only being loaded if the runmodel has passed True embedding
    embedding_matrix = np.zeros((1,0))
    #check if user prefers that embedded model is loaded with a calculated matrix weights
    if(word2vec):
        embedding_matrix = word2vec_embedding(vocab_size, tokenizer.word_index, pre_embedding, np.concatenate((data_train,data_test)))
    model = set_up_model(vocab_size, data_train_seq.shape[1], embedding_matrix)
    training = model.fit(data_train_seq, label_train, batch_size=batch_size, epochs=epochs, validation_data=(data_test_seq, label_test))

    loss_train, accuracy_train = model.evaluate(data_train_seq, label_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy_train))
    loss_test, accuracy_test = model.evaluate(data_test_seq, label_test, verbose=True)
    print("Testing Accuracy:  {:.4f}".format(accuracy_test))
    # print(data_test.shape)
    #debug trying to predict specific ones
    predictions_class = model.predict_classes(data_train_seq, batch_size)#, len(data_train_seq) // batch_size + 1 )

    print("predictions:", pd.unique(predictions_class))
    calculate_confusion_matrix(label_train, predictions_class)


    predictions = model.predict_classes(data_test_seq, batch_size)
    print("predictions:", pd.unique(predictions))
    calculate_confusion_matrix(label_test, predictions)
    return training, (accuracy_train, loss_train), (accuracy_test, loss_test)

def run_single_model(data, labels, word2vec=True, pre_embedding=False):
    #model = set_up_model()
    encoder = LabelEncoder()
    labels_int = encoder.fit(labels).transform(labels)
    print(encoder.classes_)
    label_names = encoder.classes_
    #1hot encode the labels - [1,0,0] : 0; [0,1,0]: 1; [0,0,1]: 2
    labels_encoded = utils.to_categorical(labels_int)
    data_train, data_test, label_train, label_test = train_test_split(data, labels_encoded, test_size=0.20)
    return run_model(data_train, data_test, label_train, label_test, word2vec=word2vec, pre_embedding=pre_embedding)







###helper functions###

def save_model_from_pretrained(sentences, pretrained_path, model_name, model_outpath, embedding_length):
    """will create a full model file and a word2vec file using only the words provided. This creates a smaller model than the very big sized pretrained provided
    words: list of strings"""
#embedding size must be the same as the pretrained embedding word vector length
    t = Tokenizer()
    t.fit_on_texts(sentences)
    model = Word2Vec(size=embedding_length, min_count=1)
    words = t.index_word.values()
    model.build_vocab([words], update=False)
    model.intersect_word2vec_format(pretrained_path, binary=False)
    model.save(model_outpath + model_name + ".model")
    model.wv.save_word2vec_format(model_outpath + model_name + ".wv")
    print("Model and word vectors have been saved to: {}".format(model_outpath))


def get_max_length_sentences(file_path, n_elements):
    """specific for the cw dataset. n_elements will print give out the n_longest sentences"""
    df = pd.read_csv(file_path, names=["id", "text", "label"], sep=",");
    data = df["text"].values
    data = np.array(data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    series = pd.Series(data)
    splitted = series.str.split(' ')
    max_ = list((x) for x in splitted._values)
    max_length = dict((index, len(x)) for index, x in enumerate(splitted._values))
    #max_length.sort(reverse=True)
    
    s = sorted(max_length.items(), key=lambda x: x[1], reverse=True)
    # for i in range(10):
    #     print(data[s[i][0]])
    #     print('-*10')

    return s[0:n_elements]


def get_word2vec_embedding(data):
    sentences_words = list(text_to_word_sequence(x) for x in data)
    model = Word2Vec(sentences_words, size=embedding_length, workers=4, min_count=1, sg=0, negative=5)
    
    print("Number of word vectors: {}".format(len(model.wv.vocab)))
    return model.wv

def write_df_to_file(df, pathout):
    print("writing df to {}".format(pathout))
    df.to_csv(pathout, header=False, index=False)

def clean_dict(df : DataFrame, column):
    print("Preprocessing data text - cleanining")
    d = df.duplicated(column, keep='first')
    print("removing duplicate words. Duplicates = {} words".format(len(df[d][column])))
    #drop duplicates
    print("len before removing: {}".format(len(df[column])))
    df.drop_duplicates(subset=column, inplace=True, keep='first')
    print("len after removing: {}".format(len(df[column])))
    stop = stopwords.words('english')
    for i, row in df.iterrows():
        sentence = row[column]
        #remove http url
        sentence = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', sentence)
        #remove html
        soup = BeautifulSoup(sentence, "html.parser")
        sentence = soup.get_text()
        
        #remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        
        
        #remove common stop words - might have to use it earlier because some stop words use symbols xes: isn't
        sentence = " ".join(x for x in sentence.split() if x not in stop)
        #stemming?lemming?
        #spelling correction? textblob or pyspellchecker - problem: slang and not english words will be transformed in other words.

        df.at[i, column] = sentence
    print(df[column].head())
    return df

##end helper functions


def main():
    file_path = path_full#"E:\\Users\\User\\Documents\\SCHOOL\\5thYear\\data_wrangling\\cw-partB\\train_no_dup.csv"
    df = pd.read_csv(file_path, names=["id", "text", "label"], sep=",")
    df = clean_dict(df, "text")
    data = df["text"].fillna("NAN_sentence").values
    labels = df["label"].values
    print("total labels")
    labels_n, __ = print_info_labels(labels)
    #set the embedding, otherwise is None - so wv are created randomly
    
    #training, accuracies_training , accuracies_testing = run_single_model(data, labels, word2vec=True, pre_embedding=False)
    history, accuracies_training, accuracies_testing  = run_kfold(5, data, labels, word2vec=True, pre_embedding=False)
    
    print("Training")
    print(accuracies_training)
    print("Testing")
    print(accuracies_testing)

    #print hyperparameters used
    print("max_len: {}, embedding_length : {}, filter_n : {}, filter_high : {}, strides : {}, batch_size : {}, epochs : {}".format(max_len, embedding_length, filter_n, filter_heigth, strides, batch_size, epochs))
    plt.style.use('ggplot')
    # for training in history:
    #     plot_history(training)


if __name__ == "__main__":
    main()
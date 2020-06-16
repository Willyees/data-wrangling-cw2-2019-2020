# data-wrangling-cw2-2019-2020
Second cw for data wrangling course

# Aim
Write a script on Python and by using a Convolutional Neural Network, build abusive language detection prediction model for the given dataset.
An appropriate embedding first layer has to be used along with a chosen word embedding approach.
Dataset can be filtered and cleared. The model has to be fined-tuned by modifying its parameters.

# Build
The single file Python script generated has the ability to clean and filter the dataset provided. Word vectors are ouputted by utilizing Word2Vec
and the general architecture was chosen as to be the same as “Convolutional Neural Networks for Sentence Classification” (Kim,2014).


The script can be run using different settings as: KFold, use a pretrained word vector (for example the word2vec), generate a word2vec using the datasetprovided, use random intialized word vectors.
Dataset filtering can be performed on the initial set. Otherwise it can be turned off and loaded the filtered file to save time by not clearing it at every iteration.


Helper functions are provided in order to help on setting up steps like creating filtered dictionary, save wordvector locally, generate report on the type of data labels present in the set, etc..


An [instruction](https://github.com/Willyees/data-wrangling-cw2-2019-2020/blob/master/instructions.txt) file is present, which describes the necessary actions to be performed to set up the script to be able to successfully run.
A more comprehensive analysis is provided in the [report](https://github.com/Willyees/data-wrangling-cw2-2019-2020/blob/master/report.pdf)

# Future enhancements
As explained in the instructions, some setting up must be done before script can be run. Hardcoded dataset location has be set in the script, not very elegant. This was mainly done to save time at every interaction without have to continually input it as a parameters while debugging.

In the future it would be better to expect the user to input the dataset location and the necessary hyperparamters as script parameters, rather than asking to modify the source code directly!

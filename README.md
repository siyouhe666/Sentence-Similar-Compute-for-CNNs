# Sentence-Similar-Compute-for-CNNs
I implement a paper's method use python&amp;tensorflow to measure the similarity of two sentences

Folder describe:
 model/w2v.model the word2vector model trained by gensim you can replace it.
 
File describe:
load_file_util.py Reading file and convert it to the input format, you need replace it according to your input file format.
sentences_generator.py It is just a util file, you could not mention it.
Similar_CNN.py The most important file that contains all algorithm, and you can use it to train, please read it.
Predict.py You can use it to predict new data if you have trained a model through Similar_CNN.py. By the way you'd better check the save path and read path in program.

import jieba

#Author:siyouhe666@gmail.com
#Do not worry about it, this file is just a util file.

jieba.add_word("花呗")
jieba.add_word("借呗")


class sentences_generator():
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, encoding="utf-8"):
            line = line.encode('utf-8').decode('utf-8-sig')
            sentence = line.strip().split("\t")
            sentence[1] = " ".join(jieba.cut(sentence[1]))
            sentence[2] = " ".join(jieba.cut(sentence[2]))
            yield sentence

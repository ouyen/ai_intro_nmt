from config import *
import random
# import json
import pickle

R_PATH = '.'


class Lang:

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def save(self, fpath=''):
        if fpath == '':
            fpath = R_PATH + '/dataset/' + self.name + '.pkl'
        _d = {
            'name': self.name,
            'word2index': self.word2index,
            'word2count': self.word2count,
            'index2word': self.index2word,
            'n_words': self.n_words
        }
        with open(fpath, 'wb') as f:
            pickle.dump(_d, f)

    def load(self, fpath=''):
        if fpath == '':
            fpath = R_PATH + '/dataset/' + self.name + '.pkl'
        with open(fpath, 'rb') as f:
            _d = pickle.load(f)
        self.name = _d['name']
        self.word2index = _d['word2index']
        self.word2count = _d['word2count']
        self.index2word = _d['index2word']
        self.n_words = _d['n_words']


def readLangs(lang1=LANG1, lang2=LANG2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('%s/dataset/%s2%s.txt' % (R_PATH,lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [l.split('\t') for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def tensorsFromPairs(input_lang, output_lang, pairs):
    return [tensorsFromPair(input_lang, output_lang, p) for p in pairs]


def prepareData(lang1=LANG1, lang2=LANG2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    pairs_tensors = tensorsFromPairs(input_lang, output_lang, pairs)
    return input_lang, output_lang, pairs, pairs_tensors


# def main():
#     input_lang, output_lang, pairs = prepareData(LANG1,LANG2)
#     input_lang.save()
#     output_lang.save()
#     with open(R_PATH+'/dataset/pairs.pkl','wb') as f:
#         pickle.dump(pairs,f)
#     print(random.choice(pairs))
#     return input_lang, output_lang, pairs

# def load_Langs():
#     input_lang=Lang(LANG1)
#     output_lang=Lang(LANG2)
#     input_lang.load()
#     output_lang.load()
#     with open(R_PATH+'/dataset/pairs.pkl','rb') as f:
#         pairs=pickle.load(f)
#     print(random.choice(pairs))
#     return input_lang, output_lang, pairs
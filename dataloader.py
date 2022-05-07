from torchtext.legacy import data
from config import *
import pickle


def tokenize(sentence):
    return sentence.split(' ')


class Dataloader:

    # def __init__(self,
    #              dataset_path='./dataset/',
    #              name='zh2en',
    #              lang1_file='train_zh.txt',
    #              lang2_file='train_en.txt',
    #              batch_size=TRAIN_BATCH_SIZE,
    #              save=False,
    #              load=False,
    #              load_path='',
    #              save_path=''):
    def __init__(self,
                 name=f'{LANG1}2{LANG2}',
                 type='train',
                 batch_size=TRAIN_BATCH_SIZE,
                 save=False,
                 load=False,
                 load_path=R_PATH+'/model/lang.pkl',
                 save_path=R_PATH+'/model/lang.pkl'):
        if type not in {'train', 'validation', 'test'}:
            raise 'type error'
        save = bool(type == 'train')
        load = not save
        lang1_path = f'{DATASET_PATH}{type}_{LANG1}.txt'
        lang2_path = f'{DATASET_PATH}{type}_{LANG2}.txt'

        self.name = name
        self.batch_size = batch_size
        lang1_raw_data = self.read_data(lang1_path)
        lang2_raw_data = self.read_data(lang2_path)
        # print(lang1_raw_data[0].seq)
        # print(lang2_raw_data[0].seq)
        # 返回结果
        # ['我们', '试试看', '！']
        # ['Let', 's', 'try', 'it', '.']
        if (load):
            self.load(load_path)
        else:
            self.lang1 = data.Field(init_token='<sos>',
                                    eos_token='<eos>',
                                    unk_token='<unk>',
                                    fix_length=MAX_LENGTH)
            self.lang2 = data.Field(init_token='<sos>',
                                    eos_token='<eos>',
                                    unk_token='<unk>',
                                    fix_length=MAX_LENGTH)
        fields = [("lang1", self.lang1), ("lang2", self.lang2)]
        train_examples = []
        for i in range(len(lang1_raw_data)):
            train_examples.append(
                data.Example.fromlist(
                    [lang1_raw_data[i].seq, lang2_raw_data[i].seq], fields))
        self.train_ds = data.Dataset(train_examples, fields)
        # print(self.train_ds[0].lang1)
        # print(self.train_ds[1].lang2)
        # 返回结果
        # ['我们', '试试看', '！']
        # ['Let', 's', 'have', 'a', 'try', '.']
        if not load:
            self.lang1.build_vocab(self.train_ds)
            self.lang2.build_vocab(self.train_ds)
        self.pair_tensors = data.Iterator(dataset=self.train_ds,
                                          batch_size=batch_size,
                                          device=device,
                                          train=True)
        if (save):
            self.save(save_path)
        # print(len(self.pair_tensors))
        # 返回结果
        # 19658
        # for batch in self.pair_tensors:
        #    print(batch.lang1.t())
        #    print(self.get_sentence_lang1(batch.lang1))
        #    print(self.get_sentence_lang2(batch.lang2))
        #    break
        # 返回结果
        # tensor([[   2,  636,  211,   96, 1491,  188,  636,    4,    3,    1,    1,    1,...],
        #         [   2,    9,   64,  261,   44,  259,  474, 1318,    4,    3,    1,    1,...]])
        # ['<sos> 选择 太 多 导致 无法 选择 。 <eos> <pad> <pad>...',
        #  '<sos> 他 已经 决定 要 成为 一名 教师 。 <eos> <pad> <pad>...']
        # ['<sos> Excessive choice results in the inability to choose . <eos> <pad> <pad>...',
        #  '<sos> He has decided to become a teacher . <eos> <pad> <pad>...']

    def read_data(self, path):
        SEQ = data.Field(tokenize=tokenize, sequential=True)
        return data.TabularDataset(path=path,
                                   format='csv',
                                   fields=[('seq', SEQ)])

    def save(self, path=''):
        if path == '':
            path = self.dataset_path + self.name + '.pkl'
        _d = {'name': self.name, 'lang1': self.lang1, 'lang2': self.lang2}
        with open(path, 'wb') as f:
            pickle.dump(_d, f)

    def load(self, path=''):
        if path == '':
            path = self.dataset_path + self.name + '.pkl'
        with open(path, 'rb') as f:
            _d = pickle.load(f)
        self.name = _d['name']
        self.lang1 = _d['lang1']
        self.lang2 = _d['lang2']

    def get_sentence_lang(self, lang, num_tensors):
        # num_lists = num_tensors.tolist()
        length = min(self.batch_size, num_tensors.size(1))
        res = ["" for _ in range(length)]
        langx = self.lang1 if (lang == LANG1) else self.lang2
        # for i in range(length):
        #     res.append("")
        for i in range(length):
            for j in range(num_tensors.size(0)):
                if langx.vocab.itos[num_tensors[j][i]] == '<pad>':
                    break
                res[i] += langx.vocab.itos[num_tensors[j][i]] + " "
        return res


if __name__ == '__main__':
    a = Dataloader(batch_size=5)
    # print(len(a.pair_tensors))
    for batch in a.pair_tensors:
        # print(batch.lang1.t())
        print(a.train_ds[0])
        # print(a.get_sentence_lang(LANG1,batch.lang1))
        # print(a.get_sentence_lang(LANG2,batch.lang2))

        break
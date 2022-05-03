import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple, Counter
from functools import partial

News = namedtuple('News', ['text', 'label'])

def pad_seq(seq, max_len=128):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))

class THUCNews(Dataset):

    def __init__(self, file_path):
        self.datas = THUCNews.readfile(file_path)

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)
        
    @staticmethod
    def readfile(file_path):
        datas = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                datas.append(News(text, label))
        return datas

    @staticmethod
    def build_vocab(file_paths, vocab_path='data/vocab.txt', vocab_size=6000):
        if os.path.exists(vocab_path):
            return
        datas = []
        for file_path in file_paths:
            datas.extend(THUCNews.readfile(file_path))
        chars = []
        for news in datas:
            chars.extend(list(news.text))
        counter = Counter(chars)
        # print(len(counter))
        tokens, freqs = zip(*counter.most_common(vocab_size-1))
        tokens = ['<pad>'] + list(tokens)
        with open(vocab_path, 'w', encoding='utf8') as fw:
            fw.write('\n'.join(tokens) + '\n')
    
    @classmethod
    def load_vocab(cls, vocab_path = 'data/vocab.txt'):
        if not os.path.exists(vocab_path):
            raise Exception('Vocab file don\'t exist, please invoke build_vocab previously')
        with open(vocab_path, 'r', encoding='utf8') as f:
            tokens = [line[:-1] for line in f] # 去掉最后的\n
        cls.word2id = dict(zip(tokens, range(len(tokens))))

    @classmethod
    def load_category(cls):
        categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        cls.category2id = dict(zip(categories, range(len(categories))))

    @classmethod
    def init(cls, file_paths):
        THUCNews.build_vocab(file_paths)
        THUCNews.load_category()
        THUCNews.load_vocab()
    
    @staticmethod
    def collate_fn(batch, max_len=128):
        text_ids, label_ids = [], []
        for text, label in batch:
            text_ids.append(pad_seq([THUCNews.word2id.get(token, 0) for token in text], max_len=max_len))
            label_ids.append(THUCNews.category2id[label])
        return torch.LongTensor(text_ids), torch.LongTensor(label_ids)

    @classmethod
    def vocab_size(cls):
        return len(cls.word2id)

    @classmethod
    def num_class(cls):
        return len(cls.category2id)

def get_dataloader(file_paths, batch_size=2, max_len=128):
    collate_fn = partial(THUCNews.collate_fn, max_len=max_len)
    THUCNews.init(file_paths)
    train_dataset = THUCNews(file_paths[1])
    val_dataset = THUCNews(file_paths[2])
    test_dataset = THUCNews(file_paths[0])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == '__main__':
    file_paths = ['F:\\nlp_datasets\\text_cnn\\cnews.test.txt', 'F:\\nlp_datasets\\text_cnn\\cnews.train.txt', 'F:\\nlp_datasets\\text_cnn\\cnews.val.txt']
    THUCNews.init(file_paths)
    print(THUCNews.vocab_size())
    # print(THUCNews.category2id)
    # print(list(THUCNews.word2id.items())[:10])
    # dataset = THUCNews(file_paths[0])
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=THUCNews.collate_fn)
    # for x in dataloader:
    #     print(x)
    #     break
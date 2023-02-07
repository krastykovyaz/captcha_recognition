'''module for reading data'''
from typing import List
import config
import glob
from sklearn.model_selection import train_test_split


class DataCollector:
    """
    class for preparing data
    """
    def __init__(self):
        self.captcha_img = []
        self.vocab = ['-']
        self.idx2char = {}
        self.char2idx = {}
        self.train_img = None
        self.test_img = None

    def get_files(self) ->None:
        '''
        function for reading data
        '''
        for filename in glob.glob(config.DATA_PATH + '*'):
            self.captcha_img.append(filename.split('/')[1].split('.')[0])
        self.train, self.test = train_test_split(
            self.captcha_img,
            random_state=config.RANDOM_SEED)
        return self.train, self.test


    def get_index_vocab(self) ->None:
        '''
        fuction for collect vocabulary and indexes
        '''
        self.get_files()
        images = ''.join(self.captcha_img)
        letters = sorted(set(list(images)))
        self.vocab += letters
        self.idx2char = {k: v for k,v in enumerate(self.vocab, start=0)}
        self.char2idx = {v: k for k,v in self.idx2char.items()}


if __name__=='__main__':
    reader = DataCollector()
    tr, te = reader.get_files()
    print(len(tr), len(te))
    reader.get_index_vocab()





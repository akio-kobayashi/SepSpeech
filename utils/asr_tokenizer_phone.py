import transformers
import tokenizers
from tokenizers import SentencePieceBPETokenizer
from typing import Tuple, List
import re
import json
import argparse

class ASRTokenizerBase():
    def __init__(self):
        super().__init__()
        
    def text2token(self, text) -> List[int]:
        raise NotImplementedError()
    
    def token2text_raw(self, text) -> str:
        pass
    
    def token2text(self, token) -> str:
        raise NotImplementedError()

class ASRTokenizerPhone(ASRTokenizerBase):
    def __init__(self, path, max_length=256, ctc_decode=False):
        self.blank_token = "<blk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"

        with open(path, 'r') as f:
            self.df = json.load(f)
        self._text2token = {}
        self._token2text = {}
        for key, value in self.df.items():
            self._text2token[key] = value
            self._token2text[value] = key
        self.ctc_decode = ctc_decode

        self.blank_token_id = self._text2token[self.blank_token]
        self.bos_token_id = self._text2token[self.bos_token]
        self.eos_token_id = self._text2token[self.eos_token]
        self.unk_token_id = self._text2token[self.unk_token]
        
    def text2token(self, text):
        token=[]
        token.append(self.bos_token_id)
        for phn in text.split(): # space-split sequence
            if phn == '':
                continue
            if phn not in self._text2token.keys():
                token.append(self.unk_token_id)
            else:
                token.append(self._text2token[phn])
        token.append(self.eos_token_id)
        return token

    def token2text(self, token):
        text = []
        if self.ctc_decode:
            rmvd = []
            prv_id = -1
            for id in token:
                if prv_id == id:
                    continue
                prv_id = id
                if id == self.bos_token_id or id == self.eos_token_id:
                    continue
                if id == self.blank_token_id:
                    continue
                rmvd.append(id)
                prv_id = id
        else:
            rmvd = []
            for id in token:
                if id == self.bos_token_id or id == self.eos_token_id:
                    continue
                rmvd.append(id)
        for tkn in rmvd:
            text.append(self._token2text[tkn])
        text = " ".join(text)
        return text
        
class ASRTokenizer(ASRTokenizerBase):
    def __init__(self, path, max_length=256, ctc_decode=False):
        super().__init__()
        special_tokens = ["<blk>","<s>", "</s>", "<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]
        self.tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=path,
                                                              model_max_length=max_length, 
                                                              special_tokens=special_tokens)
        self.tokenizer.blank_token = "<blk>"
        self.tokenizer.blank_token_id = self.tokenizer.encode(self.tokenizer.blank_token)[0]
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.bos_token_id = self.tokenizer.encode(self.tokenizer.bos_token)[0]
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.eos_token_id = self.tokenizer.encode(self.tokenizer.eos_token)[0]
        self.tokenizer.unk_token = "<unk>"
        self.tokenizer.cls_token = "<cls>"
        self.tokenizer.sep_token = "<sep>"
        self.tokenizer.mask_token = "<mask>"

        self.ctc_decode = ctc_decode

        self.punct = re.compile(r"⠲$")
        self.space1 = re.compile(r"^\ ")
        self.space2 = re.compile(r"\ $")
        
    def text2token(self, text) -> List[int]:
        text = re.sub(self.punct, '', text)
        text = self.tokenizer.bos_token + ' ' + text + ' '+self.tokenizer.eos_token
        return self.tokenizer.encode(text)

    def token2text_raw(self, token) -> str:
        return self.tokenizer.decode(token)
    
    def token2text(self, token) -> str:
        if self.ctc_decode:
            rmvd = []
            prv_id = -1
            for id in token:
                if prv_id == id:
                    continue
                prv_id = id
                if id == self.tokenizer.bos_token_id or id == self.tokenizer.eos_token_id:
                    continue
                if id == self.tokenizer.blank_token_id:
                    continue
                if id < 8:
                    continue
                rmvd.append(id)
                prv_id = id

            text = self.tokenizer.decode(rmvd)
        else:
            rmvd = []
            for id in token:
                if id == self.tokenizer.bos_token_id or id == self.tokenizer.eos_token_id:
                    continue
                rmvd.append(id)
            text = self.tokenizer.decode(rmvd)
        text = re.sub(self.space1, '', text)
        text = re.sub(self.space2, '', text)
        
        return text

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args=parser.parse_args()

    text = "i t a j i k i n o u e n i h i z a m a z u i t e j i b u N w a f u j o o n a n i N g e N d a t o z a N g e s u r u n o d e s U"
    tokenizer = ASRTokenizerPhone(args.path)
    token = tokenizer.text2token(text)
    text = tokenizer.token2text(token)
    print(token)
    print(text)

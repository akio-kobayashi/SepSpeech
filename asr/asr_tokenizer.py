import transformers
import tokenizers
from tokenizers import SentencePieceBPETokenizer
from typing import Tuple, List
import re

class ASRTokenizer():
    def __init__(self, path, max_length=512, ctc_decode=False, insert_space=True):
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

        self.insert_space = insert_space
        
    def text2token(self, text) -> List[int]:
        text = re.sub(self.punct, '', text)
        if self.insert_space:
            text = self.tokenizer.bos_token + ' ' + text + ' '+self.tokenizer.eos_token
        else:
            text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
            
        return self.tokenizer.encode(text)

    def token2text_raw(self, token) -> str:
        return self.tokenizer.decode(token)

    def remove(self, seq):
        temp = []
        prv_id = -1
        for id in seq:
            if prv_id == id:
                continue
            prv_id = id
            temp.append(id)
        return temp
            
    def token2text(self, token) -> str:
        max_token_id = 7
        if self.ctc_decode:
            temp = self.remove(token)
            token = [t for t in temp if t > max_token_id]
            text = self.tokenizer.decode(token)
        else:
            temp = [ t for t in token if t > max_token_id]
            text = self.tokenizer.decode(temp)
        text = re.sub(self.space1, '', text)
        text = re.sub(self.space2, '', text)
        
        return text

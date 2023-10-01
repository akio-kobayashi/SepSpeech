import transformers
import tokenizers
from tokenizers import SentencePieceBPETokenizer
from argparse import ArgumentParser
from typing import Iterator
import re

class StrIterator:
    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            self.text = [ line.strip() for line in lines ]
        self.i = 0

    def __iter__(self) -> Iterator[str]:
        return self
    
    def __next__(self):
        if self.i == len(self.text):
            raise StopIteration()
        item = self.text[self.i]
        self.i += 1
    
        return item
    
class BrailleIterator(StrIterator):
    def __init__(self, path, braille):
        super().__init__(path)
        
        assert braille is not None

        self._id2unicode={}
        with open(braille, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ary = line.strip().split()
                self._id2unicode[int(ary[2])] = ary[1]
        self.pattern1=re.compile('^(\S+)\s')
        self.pattern2=re.compile('⠲$')

    def __next__(self):
        if self.i == len(self.text):
            raise StopIteration()
        item = self.text[self.i]
        item = self.pattern2.sub('', self.pattern1.sub('', item))
        self.i += 1

        return item

class KanjiIterator(StrIterator):
    def __init__(self, path):
        super().__init__(path)
        self.pattern1=re.compile('^(\S+)\s')
        self.pattern2=re.compile('、|。|？|！|・')

    def __next__(self):
        if self.i == len(self.text):
            raise StopIteration()
        item = self.text[self.i]
        item = self.pattern2.sub('', self.pattern1.sub('',item))
        self.i += 1

        return item
    
def main(args):
    special_tokens = ["<blk>","<s>", "</s>", "<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]
    tk_tokenizer = SentencePieceBPETokenizer(add_prefix_space=False)
    if args.token_type == 'kanji':
        str_iter = KanjiIterator(args.text)
    elif args.token_type == 'braille':
        str_iter = BrailleIterator(args.text, args.braille)
    else:
        str_iter = StrIterator(args.text)
        
    tk_tokenizer.train_from_iterator(
        str_iter,
        vocab_size=4000,
        min_frequency=10,
        show_progress=True,
        special_tokens=special_tokens,
        limit_alphabet=3000
    )
    tk_tokenizer.save(args.output)

    # convert
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer, 
                                                     model_max_length=args.max_length, 
                                                     special_tokens=special_tokens)
    tokenizer.blank_token = "<blk>"
    tokenizer.blank_token_id = tk_tokenizer.token_to_id("<blk>")
    tokenizer.bos_token = "<s>"
    tokenizer.bos_token_id = tk_tokenizer.token_to_id("<s>")
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tk_tokenizer.token_to_id("<pad>")
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = tk_tokenizer.token_to_id("</s>")
    tokenizer.unk_token = "<unk>"
    tokenizer.unk_token_id = tk_tokenizer.token_to_id("<unk>")
    tokenizer.cls_token = "<cls>"
    tokenizer.cls_token_id = tk_tokenizer.token_to_id("<cls>")
    tokenizer.sep_token = "<sep>"
    tokenizer.sep_token_id = tk_tokenizer.token_to_id("<sep>")
    tokenizer.mask_token = "<mask>"
    tokenizer.mask_token_id = tk_tokenizer.token_to_id("<mask>")
    # and save for later!
    tokenizer.save_pretrained(args.pretrained)
    #print(tokenizer.encode("<s>ジャニーズ事務所のジャニー喜多川前社長から性被害を受けていたという元所属タレントの男性2人が<mask>立憲民主党の会合に出席し<mask>性被害の再発防止に向けた法整備の必要性を訴えました<mask></s>\n"))
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--output', type=str, default='tokenizer')
    parser.add_argument('--pretrained', type=str, default='pretrained')
    parser.add_argument('--max_length', type=str, default=200)
    parser.add_argument('--token_type', type=str, default='phone')
    args=parser.parse_args()

    main(args)

import transformers
import tokenizers
from tokenizers import SentencePieceBPETokenizer
from argparse import ArgumentParser
from typing import Iterator
import re

class StrIterator:
    def __init__(self, args):
        super().__init__()
        self.text = []
        for path in args.text:
            with open(path, 'r') as f:
                lines = f.readlines()
                self.text += [ line.strip() for line in lines ]
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
    def __init__(self, args):
        super().__init__(args)
        
        assert args.braille is True

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

        assert args.braille is False
        
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
    if args.braille:
        str_iter = BrailleIterator(args)
    else:
        str_iter = KanjiIterator(args)
    tk_tokenizer.train_from_iterator(
        str_iter,
        vocab_size=4000,
        min_frequency=args.min_freq,
        show_progress=True,
        special_tokens=special_tokens,
        limit_alphabet=args.limit_alphabet
    )
    #tk_tokenizer.save(args.output)

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
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text', type=str, nargs='*', required=True)
    #parser.add_argument('--output', type=str, default='tokenizer')
    parser.add_argument('--pretrained', type=str, default='pretrained')
    parser.add_argument('--max_length', type=str, default=200)
    parser.add_argument('--braille', action='store_true')
    parser.add_argument('--limit_alphabet', type=int, default=3000)
    parser.add_argument('--min_freq', type=int, default=10)
    args=parser.parse_args()

    main(args)

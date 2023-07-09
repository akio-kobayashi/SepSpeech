from asr_tokenizer import ASRTokenizer
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    args=parser.parse_args()

    tokenizer = ASRTokenizer(args.tokenizer, 256)
    with open(args.text, 'r') as f:
        line = f.readline().strip()
        tokens = tokenizer.text2token(line)
        print(tokens)
        text = tokenizer.token2text(tokens)
        print("%s" % text)

import re
import json
from argparse import ArgumentParser
from typing import Iterator
from box import Box

def main(args):

    max_length = args.max_length
    pattern = re.compile("\s")
    
    with open(args.json, 'r') as f:
        dic = json.load(f)

    mybox = Box(dic)
    vocab = mybox.model.vocab
    merges = mybox.model.merges
    
    for key in list(vocab):
        if re.search(r'<', key):
            continue
        if len(key) > max_length:
            del mybox.model.vocab[key]
    mrg=[]
    for item in merges:
        rmv = pattern.sub('', item)
        if len(rmv) <= max_length:
            mrg.append(item)
    mybox.model.merges=mrg
    dic = mybox.to_dict()
    
    with open(args.output, 'w') as f:
        json.dump(dic, f, indent=2, ensure_ascii=False)

        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--json', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=1)
    args=parser.parse_args()

    main(args)
    

import os, sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.ref, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ary = line.strip().split()
            tag = ary[-1].replace('(', '').replace(')', '')
            path = os.path.join(args.dir, tag) + '.txt'
            if os.path.exists(path):
                with open(path, 'r') as g:
                    line = ' '.join(list(g.readline().strip())) + ' (' + tag + ')'
                    print (line)
                    
if __name__ == "__main__":
    main()

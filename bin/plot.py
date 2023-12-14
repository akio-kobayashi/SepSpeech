import argparse
import json
import logging
import sys, os
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main(args):
    df = pd.read_csv(args.input_csv)
    fig, ax = plt.subplots(1, 1, dpi=300)
    ax = sns.scatterplot(data=df, x='objective', y='hasqi', hue=args.hue_name)
    ax.set_xlabel('DNSMOS (OVRL_raw)')
    ax.set_ylabel('HASQI')
    plt.savefig(args.output)
    #plt.show()
    print(df.mean(numeric_only=True))
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--hue_name', type=str, default='SNR')
    parser.add_argument('--output', type=str, default='SNR')
    args = parser.parse_args()

    main(args)


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
from sklearn.linear_model import LinearRegression

def main(args):
    df = pd.read_csv(args.input_csv)
    df[args.hue_name] = df[args.hue_name].astype(str)

    fig, ax = plt.subplots(1, 1, dpi=300)
    ax = sns.scatterplot(data=df, x='objective', y='hasqi', hue=args.hue_name)

    for snr in [0, 20]:
        model = LinearRegression()
        query = args.hue_name + '==@' + str(snr)
        temp = pd.query(query)
        df_x = temp['objective']
        df_y = temp['hasqi']
        model_lin = model.fit(df_x, df_y)
        y_fit = model_lin.predict(df_x)
        #r2_lin = model.score(df_x, df_y)
        plt.plot(df_x, y_fit, color="#000000", linewidth=0.5)
        
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


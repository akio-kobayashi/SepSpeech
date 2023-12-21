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

pesq=False
def main(args):
    df = pd.read_csv(args.input_csv)
    df[args.hue_name] = df[args.hue_name].astype(str)

    fig, ax = plt.subplots(1, 1, dpi=300)
    if pesq :
        ax.set_xlim([1,4.5])
    else:
        ax.set_xlim([1,4.5])
    ax.set_ylim([0,1])
    ax = sns.scatterplot(data=df, x='objective', y='hasqi', hue=args.hue_name, palette='gist_earth')

    '''
    r2_lins=[]    
    for snr in [0, 20, 'infty']:
        snr=str(snr)
        model = LinearRegression()
        query = args.hue_name + '==@snr'
        temp = df.query(query)
        df_x = pd.DataFrame(temp['objective'])
        df_y = pd.DataFrame(temp['hasqi'])
        model_lin = model.fit(df_x, df_y)
        y_fit = model_lin.predict(df_x)
        r2_lin = model.score(df_x, df_y)
        r2_lins.append(r2_lin)
        plt.plot(df_x.values, y_fit, color="#000000", linewidth=1.0 )
    '''
    if pesq :
        pass
        #ax.text(2.6, 0.6, '$R^{2}$='+str(round(r2_lins[0], 4)))
        #ax.text(2.3, 0.4, '$R^{2}$='+str(round(r2_lins[0], 4)))
        #ax.text(1.3, 0.3, '$R^{2}$='+str(round(r2_lins[1], 4)))
        #ax.text(2.3, 0.6, '$R^{2}$='+str(round(r2_lins[0], 4)))
        #ax.text(1.5, 0.3, '$R^{2}$='+str(round(r2_lins[1], 4)))
    else:
        pass
        #ax.text(3.5, 0.95, '$R^{2}$='+str(round(r2_lins[0], 4)))
        #ax.text(2.3, 0.1, '$R^{2}$='+str(round(r2_lins[1], 4)))
        #ax.text(3.5, 0.6, '$R^{2}$='+str(round(r2_lins[0], 4)))
        #ax.text(2.3, 0.3, '$R^{2}$='+str(round(r2_lins[1], 4)))
        
    handles, labels  =  ax.get_legend_handles_labels()
    ax.legend(handles, ['0dB', '20dB', 'inf'], loc=2, title="SNR")
    
    if pesq :
        ax.set_xlabel("PESQ")
    else:
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


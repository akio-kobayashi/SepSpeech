import numpy as np
import pandas as pd
from scipy.stats import f
import argparse
import os, sys

factors={
    'method': {'degraded', 'speaker_beam', 'e3net', 'clean_unet'},
    'snr': {'60', '20', '10', '0'},
    'packet_loss': {'0%', '5%', '10%'},
    'width': {'8kHz', '6kHz', '4kHz'}
}

def F_test(v1, v2, df1, df2):

    f_frozen = f.freeze(dfn=df1, dfd=df2)
    f_value = v1/v2
    p1 = f_frozen.sf(f_value)  # right-side
    p2 = f_frozen.cdf(f_value) # left-side

    p_value = min(p1, p2) * 2

    return f_value, p_value

def one_factor(df, factor, col_name):
    val = 0.
    for f0 in factors[factor]:
        query = factor+'==@f0'
        temp = df.query(query)
        val += np.square(temp[col_name].sum()) / len(temp)
    return val

def two_factors(df, factor1, factor2, col_name):
    val = 0.
    for f1 in factors[factor1]:
        query1 = factor1 + '==@f1'
        for f2 in factors[factor2]:
            query2 = query1 + ' & ' + factor2 + '==@f2'
            temp = df.query(query2)
            val += np.square(temp[col_name].sum()) / len(temp)
    return val

def three_factors(df, factor1, factor2, factor3, col_name):
    val = 0.
    for f1 in factors[factor1]:
        query1 = factor1 + '==@f1'
        for f2 in factors[factor2]:
            query2 = factor2 + '==@f2'
            for f3 in factors[factor3]:
                query3 = query1 + ' & ' + query2 + ' & ' + factor3 + '==@f3'
                temp = df.query(query3)
                val += np.square(temp[col_name].sum()) / len(temp)
    return val

def four_factors(df, factor1, factor2, factor3, factor4, col_name):
    val = 0.
    for f1 in factors[factor1]:
        query1 = factor1 + '==@f1'
        for f2 in factors[factor2]:
            query2 = factor2 + '==@f2'
            for f3 in factors[factor3]:
                query3 = factor3 + '==@f3'
                for f4 in factors[factor4]:
                    query4 = query1 + ' & ' + query2 + ' & ' + query3 + ' & ' + factor4 + '==@f4'
                    temp = df.query(query4)
                    val += np.square(temp[col_name].sum()) / len(temp)
    return val

def main(args):
    df = pd.read_csv(args.input_csv)
    df['snr'] = df['snr'].astype(str)
    # modified term
    CT = np.square(df[args.column_name].sum()) / len(df)

    # total square sum
    S_T = (df[args.column_name]**2).sum() - CT

    # factors
    d_A = 3
    S_A = one_factor(df, 'method', args.column_name) - CT

    d_B = 3
    S_B = one_factor(df, 'snr', args.column_name) - CT

    d_C = 2
    S_C = one_factor(df, 'packet_loss', args.column_name) - CT

    d_D = 2
    S_D = one_factor(df, 'width', args.column_name) - CT
    
    # two-factors
    d_AxB = d_A * d_B
    S_AB = two_factors(df, 'method', 'snr', args.column_name) - CT
    S_AxB = S_AB - S_A - S_B

    d_AxC = d_A * d_C
    S_AC = two_factors(df, 'method', 'packet_loss', args.column_name) - CT
    S_AxC = S_AC - S_A - S_C

    d_AxD = d_A * d_D
    S_AD = two_factors(df, 'method', 'width', args.column_name) - CT
    S_AxD = S_AD - S_A - S_D

    d_BxC = d_B * d_C
    S_BC = two_factors(df, 'snr', 'packet_loss', args.column_name) - CT
    S_BxC = S_BC - S_B - S_C

    d_BxD = d_B * d_D
    S_BD = two_factors(df, 'snr', 'width', args.column_name) - CT
    S_BxD = S_BD - S_B - S_D

    d_CxD = d_C * d_D
    S_CD = two_factors(df, 'packet_loss', 'width', args.column_name) - CT
    S_CxD = S_CD - S_C - S_D

    # three-factors
    d_AxBxC = d_A * d_B * d_C
    S_ABC = three_factors(df, 'method', 'snr', 'packet_loss', args.column_name) - CT
    S_AxBxC = S_ABC - S_A - S_B - S_C - S_AxB - S_AxC - S_BxC

    d_AxBxD = d_A * d_B * d_D
    S_ABD = three_factors(df, 'method', 'snr', 'width', args.column_name) - CT
    S_AxBxD = S_ABD - S_A - S_B - S_D - S_AxB - S_AxD - S_BxD

    d_AxCxD = d_A * d_C * d_D
    S_ACD = three_factors(df, 'method', 'packet_loss', 'width', args.column_name) - CT
    S_AxCxD = S_ACD - S_A - S_C - S_D - S_AxC - S_AxD - S_CxD

    d_BxCxD = d_B * d_C * d_D
    S_BCD = three_factors(df, 'snr', 'packet_loss', 'width', args.column_name) - CT
    S_BxCxD = S_BCD - S_B - S_C - S_D - S_BxC - S_BxD - S_CxD

    # four-factors
    d_AxBxCxD = d_A * d_B * d_C * d_D
    S_ABCD = four_factors(df, 'method', 'snr', 'packet_loss', 'width', args.column_name) - CT
    S_AxBxCxD = S_ABCD - S_A - S_B - S_C - S_D - S_AxB - S_AxC - S_AxD - S_BxC - S_BxD - S_CxD - S_AxBxC - S_AxBxD - S_AxCxD - S_BxCxD

    abcd=4*4*3*3
    r = len(df)/abcd
    d_E = abcd * (r-1)
    S_E = S_T - S_A - S_B - S_C - S_D - S_AxB - S_AxC - S_AxD - S_BxC - S_BxD - S_CxD - S_AxBxC - S_AxBxD - S_AxCxD - S_BxCxD - S_AxBxCxD

    d_T = d_A + d_B + d_C + d_AxB + d_AxC + d_AxD + d_BxC + d_BxD + d_CxD + d_AxBxC + d_AxBxD + d_AxCxD + d_BxCxD + d_AxBxCxD + d_E

    # mean square value
    V_A = S_A/d_A
    V_B = S_B/d_B
    V_C = S_D/d_D
    V_D = S_D/d_D

    V_AxB = S_AxB/d_AxB
    V_AxC = S_AxC/d_AxC
    V_AxD = S_AxD/d_AxD
    V_BxC = S_BxC/d_BxC
    V_BxD = S_BxD/d_BxD
    V_CxD = S_BxD/d_CxD

    V_AxBxC = S_AxBxC/d_AxBxC
    V_AxBxD = S_AxBxD/d_AxBxD
    V_AxCxD = S_AxCxD/d_AxCxD
    V_BxCxD = S_BxCxD/d_BxCxD

    V_AxBxCxD = S_AxBxCxD/d_AxBxCxD

    V_E = S_E/d_E

    f_A, p_A = F_test(V_A, V_E, d_A, d_E)
    f_B, p_B = F_test(V_B, V_E, d_B, d_E)
    f_C, p_C = F_test(V_C, V_E, d_C, d_E)
    f_D, p_D = F_test(V_D, V_E, d_D, d_E)

    f_AxB, p_AxB = F_test(V_AxB, V_E, d_AxB, d_E)
    f_AxC, p_AxC = F_test(V_AxC, V_E, d_AxC, d_E)
    f_AxD, p_AxD = F_test(V_AxD, V_E, d_AxD, d_E)
    f_BxC, p_BxC = F_test(V_BxC, V_E, d_BxC, d_E)
    f_BxD, p_BxD = F_test(V_BxD, V_E, d_BxD, d_E)
    f_CxD, p_CxD = F_test(V_CxD, V_E, d_CxD, d_E)

    f_AxBxC, p_AxBxC = F_test(V_AxBxC, V_E, d_AxBxC, d_E)
    f_AxBxD, p_AxBxD = F_test(V_AxBxD, V_E, d_AxBxD, d_E)
    f_AxCxD, p_AxCxD = F_test(V_AxCxD, V_E, d_AxCxD, d_E)
    f_BxCxD, p_BxCxD = F_test(V_BxCxD, V_E, d_BxCxD, d_E)

    f_AxBxCxD, p_AxBxCxD = F_test(V_AxBxCxD, V_E, d_AxBxCxD, d_E)

    print(f'A\t{f_A:.4f}\t{p_A:.4f}')
    print(f'B\t{f_B:.4f}\t{p_B:.4f}')
    print(f'C\t{f_C:.4f}\t{p_C:.4f}')
    print(f'D\t{f_D:.4f}\t{p_D:.4f}')

    print(f'AxB\t{f_AxB:.4f}\t{p_AxB:.4f}')
    print(f'AxC\t{f_AxC:.4f}\t{p_AxC:.4f}')
    print(f'AxD\t{f_AxD:.4f}\t{p_AxD:.4f}')
    print(f'BxC\t{f_BxC:.4f}\t{p_BxC:.4f}')
    print(f'BxD\t{f_BxD:.4f}\t{p_BxD:.4f}')
    print(f'CxD\t{f_CxD:.4f}\t{p_CxD:.4f}')

    print(f'AxBxC\t{f_AxBxC:.4f}\t{p_AxBxC:.4f}')
    print(f'AxBxD\t{f_AxBxD:.4f}\t{p_AxBxD:.4f}')
    print(f'AxCxD\t{f_AxCxD:.4f}\t{p_AxCxD:.4f}')
    print(f'BxCxD\t{f_BxCxD:.4f}\t{p_BxCxD:.4f}')

print(f'AxBxCxD\t{f_AxBxCxD:.4f}\t{p_AxBxCxD:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--column_name', type=str, required=True)
    args=parser.parse_args()

    main(args)

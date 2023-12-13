import pandas as pd
import argparse
import os, sys

def main(args):
    d_method, d_snr, d_width, d_packet_loss, =[], [], [], []
    d_raw_SIG, d_raw_BAK, d_raw_OVRL = [], [], []

    for csv in args.input_csv:
        if csv.startswith('mixture'):
            method='degraded'
        elif csv.startswith('tasnet'):
            method='speaker_beam'
        elif csv.startswith('e3net'):
            method='e3net'
        elif csv.startswith('unet'):
            method='clean_unet'
        else:
            raise ValueError('wrong file')
        
        df = pd.read_csv(csv)
        for idx, row in df.iterrows():
            parts = row['filename'].split('/')
                    
            if method == 'degraded':
                print(f'{parts[-1]} {parts[-5]}')
                if parts[-1].startswith('BF') and parts[-5] == 'female': continue
                if parts[-1].startswith('NF') and parts[-5] == 'female': continue

                if parts[-1].startswith('BM') and parts[-5] == 'male': continue
                if parts[-1].startswith('NM') and parts[-5] == 'male': continue
                snr=parts[-4]
                width=parts[-3]
                packet_loss=parts[-2]
            else:
                if (parts[-1].startswith('BF') or parts[-1].startswith('NF')) and parts[-3] == 'female': continue
                if (parts[-1].startswith('BM') or parts[-1].startswith('NM')) and parts[-3] == 'male': continue                
                snr=parts[-6]
                width=parts[-5]
                packet_loss=[-4]

            if snr != '0' or snr != '10' or snr != '20' or snr != '60': continue
            if snr == '60': snr='inf'                
            
            if width == '16000': width='8kHz'
            elif width == '12000': width='6kHz'
            elif width == '6000': width='4kHz'
            else:
                raise ValueError('wrong band width')
            
            if packet_loss == '0.00001': packet_loss='0%'
            elif packet_loss == '0.05': packet_loss='5%'
            elif packet_loss == '0.1': packet_loss='10%'
            else:
                raise ValueError('wrong packet loss')
            
            print(method)
            d_method.append(method)
            d_snr.append(snr)
            d_width.append(width)
            d_packet_loss.append(packet_loss)

            d_raw_SIG.append(row['raw_SIG'])
            d_raw_BAK.append(row['raw_BAK'])
            d_raw_OVRL.append(row['raw_OVRL'])
    
    df_new = pd.DataFrame()
    df_new['method'] = d_method
    df_new['snr'] = d_snr
    df_new['width'] = d_width
    df_new['packet_loss'] = d_packet_loss
    df_new['raw_SIG'] = d_raw_SIG
    df_new['raw_BAK'] = d_raw_BAK
    df_new['raw_OVRL'] = d_raw_OVRL

    df_new.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', nargs='*',)
    parser.add_argument('--output_csv', type=str, required=True)
    args=parser.parse_args()

    main(args)
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from text_generator import TextDataset
import metric
import numpy as np

def train(network, device, train_loader, optimizer, scheduler, epoch,
          iter_meter, writer):
    network.train()

    data_len = len(train_loader.dataset)

    for batch_idx, _data in enumerate(train_loader):
        input_labels, output_labels, input_lengths, output_lengths, _ = _data

        valid_lengths = [ network.valid_length(l) for l in input_lengths ]
        input_labels, output_labels = input_labels.to(device), output_labels.to(device)
        input_lengths, output_lengths = torch.tensor(valid_lengths).to(torch.int32), torch.tensor(output_lengths).to(torch.int32)
            
        input_lengths.to(device), output_lengths.to(device)

        optimizer.zero_grad()
        loss = network(input_labels, output_labels, input_lengths, output_lengths)
        loss.backward()

        if writer:
            writer.add_scalar('loss', loss.item(), iter_meter.get())
            writer.add_scalar('learning_rate', scheduler.get_lr()[0], iter_meter.get())

        optimizer.step()
        scheduler.step()
        iter_meter.step()

        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epcoh: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(input_labels), data_len,
                100. * batch_idx / len(train_loader), loss.item())
                )
        del loss
        torch.cuda.empty_cache()

def test(network, device, test_loader, epoch, iter_meter, writer):
    network.eval()

    test_loss = 0
    test_cer=[]
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            input_labels, output_labels, input_lengths, output_lengths, _ = _data

            valid_lengths = [ network.valid_length(l) for l in input_lengths ]
            input_labels, output_labels = input_labels.to(device), output_labels.to(device)
            input_lengths, output_lengths = torch.tensor(valid_lengths).to(torch.int32), torch.tensor(output_lengths).to(torch.int32)
            input_lengths.to(device), output_lengths.to(device)

            loss = network(input_labels, output_labels, input_lengths, output_lengths)
            test_loss += loss.item()/len(test_loader)

            for j in range(input_labels.shape[0]):
                pred,_ = network.greedy_decode(torch.unsqueeze(input_labels[j],0))
                #pred,_ = network.beam_search(torch.unsqueeze(input_labels[j],0))
                target = output_labels[j][:output_lengths[j]].tolist()
                c=metric.cer(target, pred)
                test_cer.append(c)

    avg_cer = sum(test_cer)/len(test_cer)
    print('Test Epcoh: {} \t Loss: {:.3f} CER: {:.3f}'.format(
        epoch, test_loss, avg_cer))

    # write logs
    if writer:
        writer.add_scalar('test_loss', test_loss, iter_meter.get())
        writer.add_scalar('cer', avg_cer, iter_meter.get())

    network.train()
    return avg_cer

def decode(network, device, test_loader, tokenizer, output_tokenizer, outpath, beam_search=False):
    network.eval()

    test_loss = 0
    test_cer=[]
    n=0

    with open(outpath, 'w') as f:
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                input_labels, output_labels, input_lengths, output_lengths, keys = _data

                valid_lengths = [ network.valid_length(l) for l in input_lengths ]

                input_labels, output_labels = input_labels.to(device), output_labels.to(device)
                input_lengths, output_lengths = torch.tensor(valid_lengths).to(torch.int32), torch.tensor(output_lengths).to(torch.int32)
                input_lengths.to(device), output_lengths.to(device)

                xs = network.ff_encoder(input_labels)
                for j in range(xs.shape[0]):
                    if beam_search:
                        pred, logp = network.beam_search(torch.unsqueeze(xs[j],0), ff=False)
                    else:
                        pred, logp = network.greedy_decode(torch.unsqueeze(xs[j],0), ff=False)

                    target = output_labels[j][:output_lengths[j]].tolist()
                    c=metric.cer(target, pred)
                    test_cer.append(c)
                
                    output = output_tokenizer.token2text(pred)
                    output = ''.join(list(output))
                
                    f.write(f'{output} ({keys[j]})\n')

    avg_cer = sum(test_cer)/len(test_cer)
    network.train()
    return avg_cer

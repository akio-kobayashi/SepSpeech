import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from generator import SpeechDataset
import metric
import numpy as np

def train(network, device, train_loader, optimizer, scheduler, epoch,
            iter_meter, writer):
    network.train()

    data_len = len(train_loader.dataset)

    for batch_idx, _data in enumerate(train_loader):
        inputs, labels, input_lengths, label_lengths, _ = _data

        input_lengths = network.valid_input_lengths(input_lengths)
        
        inputs, labels = inputs.to(device), labels.to(device)
        input_lengths, label_lengths = torch.tensor(input_lengths).to(torch.int32), torch.tensor(label_lengths).to(torch.int32)
        input_lengths.to(device), label_lengths.to(device)

        optimizer.zero_grad()
        loss = network(inputs, labels, input_lengths, label_lengths)
        loss.backward()

        if writer:
            writer.add_scalar('loss', loss.item(), iter_meter.get())
            writer.add_scalar('learning_rate', scheduler.get_lr()[0], iter_meter.get())

        optimizer.step()
        scheduler.step()
        iter_meter.step()

        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epcoh: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), data_len,
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
            inputs, labels, input_lengths, label_lengths, _ = _data

            input_lengths = network.valid_input_lengths(input_lengths)
            
            inputs, labels = inputs.to(device), labels.to(device)
            input_lengths, label_lengths = torch.tensor(input_lengths).to(torch.int32), torch.tensor(label_lengths).to(torch.int32)
            input_lengths.to(device), label_lengths.to(device)

            loss = network(inputs, labels, input_lengths, label_lengths)
            test_loss += loss.item()/len(test_loader)

            for j in range(inputs.shape[0]):
                pred,_ = network.greedy_decode(torch.unsqueeze(inputs[j],0))
                target = labels[j][:label_lengths[j]].tolist()
                c=metric.cer(target, pred)
                test_cer.append(c)

    avg_cer = sum(test_cer)/len(test_cer)
    print('Test Epcoh: {} \t Loss: {:.3f} CER: {:.3f}'.format(
        epoch, test_loss, avg_cer))

    # write logs
    if writer:
        writer.add_scalar('test_loss', test_loss, iter_meter.get())
        writer.add_scalar('cer', avg_cer, iter_meter.get())

    return avg_cer

def decode(network, device, test_loader, tokenizer, outpath, beam_search=False):
    network.eval()

    test_loss = 0
    test_cer=[]
    n=0

    with open(outpath, 'w') as f:
        with torch.no_grad():
            for i, _data in enumerate(test_loader):
                inputs, labels, input_lengths, label_lengths, keys = _data

                input_lengths = network.valid_input_lengths(input_lengths)
                inputs, labels = inputs.to(device), labels.to(device)
                input_lengths, label_lengths = torch.tensor(input_lengths).to(torch.int32), torch.tensor(label_lengths).to(torch.int32)
                input_lengths.to(device), label_lengths.to(device)

                xs = network.ff_encoder(inputs)
                for j in range(xs.shape[0]):
                    if beam_search:
                        pred, logp = network.beam_search(torch.unsqueeze(xs[j],0), ff=False)
                    else:
                        pred, logp = network.greedy_decode(torch.unsqueeze(xs[j],0), ff=False)

                    target = labels[j][:label_lengths[j]].tolist()
                    c=metric.cer(target, pred)
                    test_cer.append(c)
                
                    output = tokenizer.token2text(pred)
                    output = ' '.join(list(output))
                
                    f.write(f'{output} ({keys[j]})\n')

    avg_cer = sum(test_cer)/len(test_cer)
    return avg_cer

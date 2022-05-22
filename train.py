from lib2to3.pgen2.token import RPAR
from config import *

# from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import time
import math
from model import *

# R_PATH = '..'
from dataloader import Dataloader
# from data_processing import prepareData
# input_lang, output_lang, pairs=load_Langs()
# input_lang, output_lang, pairs, pairs_tensors = prepareData(LANG1, LANG2)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# def train(input_tensor,
#           target_tensor,
#           seq_model:Seq2seq,
#           encoder_optimizer,
#           decoder_optimizer,
#           loss_fn,
#           max_length=MAX_LENGTH):

#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#     model_output = seq_model(input_tensor)
#     loss=loss_fn(model_output,target_tensor)
#     loss.backward()
#     encoder_optimizer.step()
#     decoder_optimizer.step()
#     return loss

# def loss_fn(model_output,target):
#     loss=0
#     criterion = nn.NLLLoss()
#     for i in range(MAX_LENGTH):

teacher_forcing_ratio = TEACHER_FORCING_RATIO


def train(input_tensor,
          target_tensor,
          encoder,
          decoder,
          encoder_optimizer,
          decoder_optimizer,
          criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length,
                                  encoder.hidden_size,
                                  device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = bool(random.random() < teacher_forcing_ratio)

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di].view(1))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach(
            )  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].view(1))
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(
    dataloader: Dataloader,
    dataloader_validation:Dataloader,
    encoder,
    decoder,
    #    n_iters,
    #    print_every=1000,
    #    plot_every=100,
    learning_rate=0.01):
    start = time.time()
    losses = []
    bleu_scores = []
    max_bleu=0
    loss_total = 0
    s2s=Seq2seq(encoder,decoder)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    training_pairs = dataloader.pair_tensors
    criterion = nn.NLLLoss()

    # for iter in range(1, TRAIN_BATCH_SIZE + 1):
    batch_count = 0

    for batch in dataloader.pair_tensors:
        if batch.lang1.size()[-1]!=TRAIN_BATCH_SIZE:
            break
        batch_count += 1
        # training_pair = training_pairs[iter - 1]
        loss_total = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0
        length_total = 0

        for iter in range(TRAIN_BATCH_SIZE):
            input_tensor = batch.lang1[:, iter]
            target_tensor = batch.lang2[:, iter]


            decoder_outputs, output_length = s2s(input_tensor)
            # loss=0
            length_total+=output_length
            # for i in range(output_length):
            #     loss+=criterion(decoder_outputs[i],target_tensor[i].view(1))
            loss+=criterion(decoder_outputs,target_tensor)


            # loss = train(input_tensor, target_tensor, encoder, decoder,
            #              encoder_optimizer, decoder_optimizer, criterion)
            # print_loss_total += loss
            # plot_loss_total += loss

        loss.backward()


        encoder_optimizer.step()
        decoder_optimizer.step()
        # if iter % print_every == 0:
        # loss_avg = loss.item() / (TRAIN_BATCH_SIZE*length_total)
        loss_avg = loss.item() /TRAIN_BATCH_SIZE
        losses.append(loss_avg)

        print('%s (%d %d%%) %.4f' %
              (timeSince(start, batch_count * TRAIN_BATCH_SIZE /
                         TOTAL_TRAIN_SIZE), batch_count, batch_count *
               TRAIN_BATCH_SIZE / TOTAL_TRAIN_SIZE * 100, loss_avg))
        if batch_count%BLEU_BATCH_COUNT==0:
            b=dataloader_validation.bleu(s2s)
            b_score=sum(b)/len(b)
            bleu_scores.append(b_score)
            if b_score>max_bleu:
                torch.save(encoder.state_dict(), R_PATH + "/model/encoder1.pt")
                torch.save(decoder.state_dict(), R_PATH + "/model/attn_decoder1.pt")
                torch.save(s2s.state_dict(), R_PATH+'/model/s2s.pt')
                max_bleu=b_score
        # if iter % plot_every == 0:
        # plot_loss_avg = plot_loss_total / plot_every
        # plot_losses.append(plot_loss_avg)
        # plot_loss_total = 0

    # showPlot(plot_losses)
    return losses, bleu_scores


def train_and_save():
    hidden_size = 256
    dataloader = Dataloader(type='train')
    dataloader_validation = Dataloader(type='validation')
    encoder1 = EncoderRNN(len(dataloader.lang1.vocab.itos),
                          hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size,
                                   len(dataloader.lang2.vocab.itos),
                                   dropout_p=0.1).to(device)

    losses, bleu_scores = trainIters(dataloader,dataloader_validation, encoder1, attn_decoder1)

    t_plot_losses = torch.tensor(losses)
    t_bleu_scores = torch.tensor(bleu_scores)
    torch.save(t_plot_losses, R_PATH + "/model/plot_losses.pt")
    torch.save(t_bleu_scores, R_PATH + "/model/bleu_scores.pt")
    try:
        from message import message
        message()
    except:
        pass


if __name__ == '__main__':
    train_and_save()

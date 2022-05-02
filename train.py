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

R_PATH = '..'
from data_processing import prepareData
# input_lang, output_lang, pairs=load_Langs()
input_lang, output_lang, pairs, pairs_tensors = prepareData(LANG1, LANG2)


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
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach(
            )  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder,
               decoder,
               n_iters,
               print_every=1000,
               plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    training_pairs = [random.choice(pairs_tensors) for _ in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' %
                  (timeSince(start, iter / n_iters), iter,
                   iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)
    return plot_losses


def train_and_save():
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size,
                                   output_lang.n_words,
                                   dropout_p=0.1).to(device)

    plot_losses = trainIters(encoder1, attn_decoder1, 75000, print_every=500)
    torch.save(encoder1.state_dict(), R_PATH + "/model/encoder1.pt")
    torch.save(attn_decoder1.state_dict(), R_PATH + "/model/attn_decoder1.pt")

    t_plot_losses = torch.tensor(plot_losses)
    torch.save(t_plot_losses, R_PATH + "/model/plot_losses.pt")


if __name__ == '__main__':
    train_and_save()

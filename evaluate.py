from config import *

# from data_processing import load_Langs,Lang

# input_lang, output_lang, pairs=load_Langs()

from model import *
# from train import tensorFromSentence
# from data_processing import tensorFromSentence
from dataloader import Dataloader
import matplotlib.pyplot as plt
from torchtext.data.metrics import bleu_score


def cal_bleu(
    dataloader: Dataloader,
    input_sentences,
    target_sentences,
    input_tensors,
    encoder,
    decoder,
):
    # list(a.train_ds.lang1)
    output_sentences=[[]]
    b=bleu_score(input_sentences,output_sentences,)


def evaluate(dataloader: Dataloader,
             pairs,
             encoder,
             decoder,
             sentence,
             input_tensor='',
             max_length=MAX_LENGTH):
    with torch.no_grad():
        # input_tensor = tensorFromSentence(input_lang, sentence)
        if input_tensor=='':
            input_tensor = dataloader.index_sentences(LANG1, [sentence]).T[0]
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length,
                                      encoder.hidden_size,
                                      device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(dataloader.lang2.vocab.itos[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(input_lang, output_lang, pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(input_lang, output_lang, pairs,
                                            encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    encoder1_sd = torch.load('../model/encoder1.pt', map_location='cpu')
    encoder1.load_state_dict(encoder1_sd)

    attn_decoder1 = AttnDecoderRNN(hidden_size,
                                   output_lang.n_words,
                                   dropout_p=0.1).to(device)
    attn_decoder1_sd = torch.load('../model/attn_decoder1.pt',
                                  map_location='cpu')
    attn_decoder1.load_state_dict(attn_decoder1_sd)
    # trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
    plot_losses = torch.load('../model/plot_losses.pt', map_location='cpu')
    plt.plot(plot_losses)
    plt.savefig('loss.jpg', dpi=400)

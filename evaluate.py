from config import *

# from data_processing import load_Langs,Lang

# input_lang, output_lang, pairs=load_Langs()

from model import *
# from train import tensorFromSentence
from data_processing import tensorFromSentence
import matplotlib.pyplot as plt
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def evaluate(input_lang,
             output_lang,
             pairs,
             encoder,
             decoder,
             sentence,
             max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
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
                # decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_bleu(input_lang, output_lang, pairs, encoder, decoder, type='',n=1024,show=False):
    p_list = []
    b_list = []
    if type=='val':
        p=pairs[-2048:-1024]
    elif type=='test':
        p=pairs[-1024:]
    for i in range(n):
        # pair = random.choice(pairs)
        pair = p[i]
        output_words, attentions = evaluate(input_lang, output_lang, pairs,
                                            encoder, decoder, pair[0])
        target_words = pair[1].split(' ')
        output_sentence = ' '.join(output_words)
        b = sentence_bleu([target_words], output_words, weights=(0.5, 0.5, 0., 0.),smoothing_function = SmoothingFunction().method1)
        # b = bleu_score([output_words],[[target_words]],max_n=2)
        if show:
            print('>', pair[0])
            print('=', pair[1])
            print('<', output_sentence)
            print('bleu: ',b)
            print('')
        b_list.append(b)
        p_list.append((pair[0],pair[1],output_sentence))
    return p_list,b_list
        


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

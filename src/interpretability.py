import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import selfies as sf
import sentencepiece as spm
import torch

from .parameters import *
from .utils import *

def attribution(args, model, input_sentence, device, n_step=50):
    from torch.autograd import Variable, grad
    
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{args.src_sp_prefix}.model")
    trg_sp.Load(f"{args.trg_sp_prefix}.model")
    
    tokenized = src_sp.EncodeAsIds(input_sentence)
    src_data = torch.LongTensor(pad_or_truncate(tokenized, args.src_seq_len)).unsqueeze(0).to(device) # (1, L)
    e_mask = (src_data != pad_id).unsqueeze(1).to(device) # (1, 1, L)
    
    IG = []
    model.eval()
    for n in range(1, n_step+1):
        # move the input along the path from alpha=0 to alpha=1
        alpha = float(n/n_step)
        src_embedding = model.src_embedding(src_data)
        src_embedding = model.src_positional_encoder(src_embedding)      
        
        baseline = torch.zeros(src_embedding.shape).to(device)
        difference = src_embedding - baseline
            
        encoder_input = baseline + alpha * (difference)
        encoder_output = model.encoder(encoder_input, e_mask)
        
        # Decoding...
        last_words = torch.LongTensor([pad_id] * args.trg_seq_len).to(device) # (L)
        last_words[0] = sos_id # (L)
        cur_len = 1
        for i in range(trg_seq_len):
            d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
            nopeak_mask = torch.ones([1,  args.trg_seq_len,  args.trg_seq_len], dtype=torch.bool).to(device)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

            trg_embedded = model.trg_embedding(last_words)
            trg_positional_encoded = model.trg_positional_encoder(trg_embedded)
            decoder_output, attn_weight = model.decoder(
                trg_positional_encoded,
                encoder_output,
                e_mask,
                d_mask
            ) # (1, L, d_model)

            lin_output = model.output_linear(decoder_output)
            output = model.softmax(
                model.output_linear(decoder_output)
            ) # (1, L, trg_vocab_size)

            output = torch.argmax(output, dim=-1) # (1, L)
            last_word_id = output[0][i].item()

            if i < trg_seq_len-1:
                last_words[i+1] = last_word_id
                cur_len += 1

            if last_word_id == eos_id:
                break
        #----------------------------- End of decoding
        #print(f'step:{n} -> { trg_sp.decode_ids(last_words.tolist())}')
        
        # compute gradients
        vals, ids = torch.max(lin_output, -1)    
        for id_, translated_word in enumerate(vals.squeeze()):
            if n == 1:
                IG.append( torch.sum(1/n_step * difference * grad(translated_word, src_embedding, retain_graph=True,allow_unused=True)[0], dim=-1))
            else:
                IG[id_] += torch.sum(1/n_step * difference * grad(translated_word, src_embedding, retain_graph=True,allow_unused=True)[0], dim=-1)
                
    # remove pad_tokens
    ig_grid = []
    for token, i in zip(last_words.tolist(), IG):
        if token == 2:
            break
        ig_grid.append(i.squeeze().detach().cpu().numpy()[:len(input_sentence.split())])

    return {"IG":np.array(ig_grid)[:-1], #IG,
           "src": input_sentence,
            "src_tokens": tokenized,
           "pred_tokens": last_words.tolist(),
           "pred":  trg_sp.decode_ids(last_words.tolist())}



def IGplot(results, 
           input_type,
           output_type,
           savefig = None,
           title='Integrated Gradient Visualisation',
           cmap_color = 'PuOr',
           return_fig = False):

    src = results['src']
    predicted = results['pred']
    ig_grid = results['IG']
    
    fig, axs = plt.subplots(nrows=2, ncols=1, 
                            gridspec_kw={'height_ratios': [10, 1], }, 
                            figsize=(10, 10), dpi=400, facecolor='white')

    axh = sns.heatmap(ig_grid, 
                      yticklabels = predicted.split(),
                      xticklabels = [None for i in src.split()],
                      annot=None, fmt="s", 
                      ax=axs[0],
                      vmax =  1,#np.max(ig_grid, ), 
                      vmin = -1,
                      linewidth=0.6, square=False, 
                      cmap= cmap_color, #'BuPu', #'RdBu',#'inferno'
                      cbar_kws={ 'location':'top', 'orientation':'horizontal'}
                     )
    axh.figure.axes[-1].xaxis.label.set_size(13)

    #summation
    sns.heatmap((np.expand_dims(ig_grid.sum(axis=0).round(2), axis=0)), 
                annot=True, 
                yticklabels = ['Total'],
                xticklabels = src.split(),
                linewidth=0.6,  
                vmax =  1,#np.max(ig_grid, ), 
                vmin = -1,
                ax=axs[1], cbar=False, square=True, 
                cmap=cmap_color,
                cbar_kws = dict(use_gridspec=True,orientation="horizontal"))
    axs[1].set_title('Total weights', fontsize = 13)
    #-------end of summation


    fig.suptitle(title, fontsize = 15)

    # Set common labels
    axs[0].set_ylabel(output_type, fontsize = 13)
    axs[1].set_xlabel(input_type, fontsize = 13)
    # axs.set_yticks(list(range(len(predicted.split()))), labels=predicted.split())
    # ax.tick_params(axis='both', which='major',  labelbottom = False, bottom=False, top = True, labeltop=True)

    fig.tight_layout()

    if savefig:
        fig.savefig(savefig)
        
    if return_fig:
        return fig
    
    
def Attn_plot(src, predicted, attn, input_type, output_type, savefig=None, cmap_color='Purples', return_fig=False):
    fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [10, 2], }, figsize=(10, 10), dpi=400, facecolor='white')
    axh = sns.heatmap(attn, 
                      yticklabels = predicted.split(),
                      xticklabels = [None for i in src.split()],
                      annot=None, fmt="s", 
                      ax=axs[0], vmax = 1, linewidth=0.6, square=False, 
                      cmap= cmap_color,#'inferno'
                      cbar_kws={ 'location':'top', 'orientation':'horizontal'}
                     )
    axh.figure.axes[-1].xaxis.label.set_size(13)
    sns.heatmap((np.expand_dims(attn.sum(axis=0).round(2), axis=0)), 
                annot=True, 
                yticklabels = ['Total'],
                xticklabels = src.split(),
                linewidth=0.6,  
                ax=axs[1], cbar=False, square=True, 
                cmap=cmap_color,
                cbar_kws = dict(use_gridspec=True,orientation="horizontal"))
    axs[1].set_title('Total attention weights', fontsize = 13)
    fig.suptitle('Attention Visualisation', fontsize = 15)

    # Set common labels
    axs[0].set_ylabel(output_type, fontsize = 13)
    axs[1].set_xlabel(input_type, fontsize = 13)
    # axs.set_yticks(list(range(len(predicted.split()))), labels=predicted.split())
    # axs[0].tick_params(axis='x', which='major',  labelbottom = False, bottom=False, top = False, labeltop=False)

    fig.tight_layout()
    if savefig:
        fig.savefig(savefig)
    
    if return_fig:
        return fig

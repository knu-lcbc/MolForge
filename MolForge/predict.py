import argparse
import copy
import datetime
import heapq
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger
import selfies as sf
import sentencepiece as spm

import torch
import torch.nn as nn

from .parameters import *
from .utils import *
from .transformer import *
from .decoder import greedy_search, beam_search


# Use this python module for inference mode


def setup(model, checkpoint_path, args, print_=True):
    if print_: print("Loading checkpoint...", args.fp, args.model_type)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.rank))
    updated_checkpoint= {}
    for key, val in checkpoint['model_state_dict'].items():
        if 'module.' in key:
            updated_checkpoint[key.replace('module.', '')] = val
        else:
            updated_checkpoint[key] = val

    model.load_state_dict(updated_checkpoint)

    return model.to(args.rank)


def inference(model, input_sentence, method, args, return_attn=False):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{args.src_sp_prefix}.model")
    trg_sp.Load(f"{args.trg_sp_prefix}.model")

    print("Preprocessing input sentence...")
    tokenized = src_sp.EncodeAsIds(input_sentence)
    src_data = torch.LongTensor(pad_or_truncate(tokenized, args.src_seq_len)).unsqueeze(0).to(args.device) # (1, L)
    e_mask = (src_data != pad_id).unsqueeze(1).to(args.device) # (1, 1, L)

    start_time = datetime.datetime.now()

    print("Encoding input sentence...")
    model.eval()
    src_data = model.src_embedding(src_data)
    src_data = model.src_positional_encoder(src_data)
    e_output = model.encoder(src_data, e_mask) # (1, L, d_model)

    if method == 'greedy':
        print("Greedy decoding selected.")
        result, attn = greedy_search(model, e_output, e_mask, trg_sp, args.device, True)
    elif method == 'beam':
        print("Beam search selected.")
        result, attn = beam_search(model, e_output, e_mask, trg_sp, args.device, True)
    print()

    end_time = datetime.datetime.now()

    total_inference_time = end_time - start_time
    seconds = total_inference_time.seconds
    minutes = seconds // 60
    seconds = seconds % 60

    print(f"Input: {input_sentence}")
    print(f"Result: {result}")
    print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")

    if return_attn:
        return result, attn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', '--fingerprint', required=True, help='Name of the fingerprint')
    parser.add_argument('--model_type', required=True, help='The representation of molecules, either "smiles" or "selfies"')

    parser.add_argument('--input', type=str, help='An input sequence')
    parser.add_argument('--input_file', '--file', metavar='PATH', help="Read input sequence from a file")
    parser.add_argument('--checkpoint', type=str, help="Checkpoint file for given model.")
    parser.add_argument('--decode', type=str, default='greedy', help="greedy or beam?")


    args = parser.parse_args()
    # checking the settings...
    assert args.fp in fp_names, f"Choose one the fingerprints: \n{fp_names}"
    assert args.model_type in ['smiles', 'selfies'], f"Enter either 'smiles' or 'selfies'"

    if args.input_file:
        assert os.path.exists(args.input_file), f"There is no input file name {args.input_file}."
    else:
        assert args.input, f"Please provide an input sequence via 'input' or 'input_file' argument."

    if args.checkpoint:
        assert os.path.exists(f"{ckpt_dir}/{args.checkpoint}"), f"There is no checkpoint named {args.checkpoint}."
        args.checkpoint = f"{ckpt_dir}/{args.checkpoint}"
    else:
        args.checkpoint = f"{ckpt_dir}/{args.fp}_{args.model_type}_checkpoint.pth"
        assert os.path.exists(f"{args.checkpoint}"), f"There is no checkpoint {args.checkpoint} in the saved_models."

    assert args.decode == 'greedy' or args.decode =='beam', "Please specify correct decoding method, either 'greedy' or 'beam'."

    args.src_vocab_size = fp_vocab_sizes[args.fp]
    args.trg_vocab_size = trg_vocab_sizes[args.model_type]
    args.src_seq_len = fp_seq_lens[args.fp]
    args.trg_seq_len = trg_seq_len

    args.root_dir = root_dir
    args.fp_datadir = data_dir.joinpath('fingerprints', args.fp)

    args.src_sp_prefix = f"{sp_dir}/{args.fp}_{SP_NAME}"
    args.trg_sp_prefix = f"{sp_dir}/{args.model_type}_{SP_NAME}"

    args.rank  = torch.device('cuda')  if torch.cuda.is_available() else torch.device('cpu')
    args.device = args.rank

    print('Here we go..')
    [ print(f'{i :>15} : {j}') for i,j in vars(args).items()]
    print(f"")

    model = setup(build_model(args).to(args.device), args.checkpoint, args)

    inference(model, args.input, args.decode, args)

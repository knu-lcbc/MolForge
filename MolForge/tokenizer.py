import argparse
import copy
import datetime
from io import StringIO
import heapq
import os
from pprint import pprint
import sys
import random
import shutil

import sentencepiece as spm

from .parameters import *


def train_sp(args, split=False):
    template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}"

    if args.dataset:
        src_file = '/tmp/tmp.src_file'
        trg_file = '/tmp/.trg_file'
        with open(src_file, 'w') as src_f, open(trg_file, 'w') as trg_f:
            for line in open(args.dataset):
                trg, src = line.strip().split('\t')
                src_f.write(f'{src}\n')
                trg_f.write(f'{trg}\n')

    else:
        src_file = '/tmp/tmp.src_file'
        trg_file = '/tmp/.trg_file'
        with open(src_file, 'w') as src_f, open(trg_file, 'w') as trg_f:
            for line in open(data_dir.joinpath('fingerprints', f'{args.fp}.{args.model_type}.{TRAIN_DATA}')):
                trg, src = line.strip().split('\t')
                src_f.write(f'{src}\n')
                trg_f.write(f'{trg}\n')

            for line in open(data_dir.joinpath('fingerprints', f'{args.fp}.{args.model_type}.{VALID_DATA}')):
                trg, src = line.strip().split('\t')
                src_f.write(f'{src}\n')
                trg_f.write(f'{trg}\n')

    for input_file, output_prefix, vocab_size in [[src_file, args.src_sp_prefix, args.src_vocab_size],
                                          [trg_file, args.trg_sp_prefix, args.trg_vocab_size]]:
        config = template.format(input_file,
                                pad_id,
                                sos_id,
                                eos_id,
                                unk_id,
                                output_prefix,
                                vocab_size,
                                character_coverage,
                                sp_model_type)

        print(config)

        if not os.path.isdir(sp_dir):
            os.mkdir(sp_dir)

        print(spm)
        spm.SentencePieceTrainer.Train(config)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', required=True, help='A name of the fingerprint')
    parser.add_argument('--model_type', required=True, help='The representation of molecules, either "smiles" or "selfies"')
    parser.add_argument('--dataset', help='Raw dataset file which has smiles and fingerprint (tab separated)')



    args = parser.parse_args()
    # checking the settings...
    assert args.fp in fp_names, f"Choose one the fingerprints: \n{fp_names}"
    assert args.model_type in ['smiles', 'selfies'], f"Enter either 'smiles' or 'selfies'"

    args.src_vocab_size = fp_vocab_sizes[args.fp]
    args.trg_vocab_size = trg_vocab_sizes[args.model_type]

    args.root_dir = root_dir
    args.fp_datadir = data_dir.joinpath('fingerprints', args.fp)

    args.src_sp_prefix = f"{sp_dir}/{args.fp}_{SP_NAME}"
    args.trg_sp_prefix = f"{sp_dir}/{args.model_type}_{SP_NAME}"


    print('Here we go..')
    [ print(f'{i :>15} : {j}') for i,j in vars(args).items()]
    print(f"")

    train_sp(args, )

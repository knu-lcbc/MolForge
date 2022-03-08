import argparse
import copy
import datetime
import heapq
import os
from pathlib import Path
import re
import random
import shutil
import sys

from .parameters import *
from .utils import *
from .transformer import *
from .decoder import greedy_search, beam_search

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import selfies as sf
import sentencepiece as spm
from rdkit import Chem, RDLogger



def setup(model, checkpoint_path, args):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.rank))
    updated_checkpoint= {}
    for key, val in checkpoint['model_state_dict'].items():
        if 'module.' in key:
            updated_checkpoint[key.replace('module.', '')] = val
        else:
            updated_checkpoint[key] = val

    model.load_state_dict(updated_checkpoint)
    print('Checkpoint is successfully loaded!')
    return model.to(args.rank)


def evaluate(args, model, model_type, valid_loader, device,  method='greedy'):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{args.src_sp_prefix}.model")
    trg_sp.Load(f"{args.trg_sp_prefix}.model")

    print('Start predicting...')
    start_time = datetime.datetime.now()
    tanimoto_similarities = []
    total,exact, _100, _90, _85, _80, _70, _60, _50, _40, _30, _20, _00, _invalid = 0,0,0, 0,0,0,0,0, 0, 0, 0, 0,0, 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            src_input, trg_input, trg_output = batch
            src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

            #e_mask, d_mask = make_mask(src_input, trg_input)
            for j in  range(len(src_input)):
                # preparing src data for encoder
                src_j = src_input[j].unsqueeze(0).to(device) # (L) => (1, L)
                encoder_mask = (src_j != pad_id).unsqueeze(1).to(device) # (1, L) => (1, 1, L)
                # encoding src input
                src_j = model.src_embedding(src_j) # (1, L) => (1, L, d_model)
                src_j = model.src_positional_encoder(src_j) # (1, L, d_model)
                encoder_output = model.encoder(src_j, encoder_mask) # (1, L, d_model)
                if method == 'greedy':
                    s_pred = greedy_search(model, encoder_output, encoder_mask, trg_sp, device)

                    s_src   = src_sp.decode_ids(src_input[j].tolist())
                    s_truth = trg_sp.decode_ids(trg_output[j].tolist())

                    truth_smi, pred_smi = s_truth.replace(' ' , ''), s_pred.replace(' ', '')
                    if model_type == 'selfies':
                        RDLogger.DisableLog('rdApp.*')

                        try:
                            truth_smi = Chem.CanonSmiles(sf.decoder(truth_smi))
                            pred_smi = Chem.CanonSmiles(sf.decoder(pred_smi))
                        except:
                            truth_smi = sf.decoder(truth_smi)
                            pred_smi = sf.decoder(pred_smi)

                    #print(f'{truth_smi}|{pred_smi}|{s_src}\n')

                    tanimoto = morganfpTc(truth_smi, pred_smi )
                    #tanimoto = smiles_fp_similarity(truth_smi, pred_smi )
                    print(f"Prediction|{tanimoto}|{s_truth.replace(' ', '')}|{s_pred.replace(' ', '')}|{s_src}")
                    total += 1
                    if truth_smi == pred_smi:
                        exact += 1
                    if tanimoto == 1.0:
                        _100 += 1
                    elif tanimoto >= 0.90:
                        _90 += 1
                    elif tanimoto >= 0.85:
                        _85 += 1
                    elif tanimoto >= 0.80:
                        _80 += 1
                    elif tanimoto >= 0.70:
                        _70 += 1
                    elif tanimoto >= 0.60:
                        _60 += 1
                    elif tanimoto >= 0.50:
                        _50 += 1
                    elif tanimoto >= 0.40:
                        _40 += 1
                    elif tanimoto >= 0.30:
                        _30 += 1
                    elif tanimoto >= 0.20:
                        _20 += 1
                    elif tanimoto > 0.0:
                        _00 += 1
                    else:
                        _invalid += 1

                    tanimoto_similarities.append(tanimoto)

                elif method == 'beam':
                    s_preds, sscores = beam_search(model, encoder_output, encoder_mask, trg_sp, device, return_candidates=True)

                    s_src   = src_sp.decode_ids(src_input[j].tolist())
                    s_truth = trg_sp.decode_ids(trg_output[j].tolist())
                    truth_smi = s_truth.replace(' ' , '')
                    if model_type == 'selfies':
                        RDLogger.DisableLog('rdApp.*')
                        try:
                            truth_smi = Chem.CanonSmiles(sf.decoder(truth_smi))
                        except:
                            truth_smi = sf.decoder(truth_smi)

                    print(f"Prediction|{s_truth.replace(' ', '')}", end='')
                    tanimoto = 0
                    spred = None
                    for  s_pred in s_preds:
                        pred_smi = s_pred.replace(' ' ,'')
                        if model_type == 'selfies':
                            RDLogger.DisableLog('rdApp.*')
                            try:
                                pred_smi = Chem.CanonSmiles(sf.decoder(pred_smi))
                            except:
                                pred_smi = sf.decoder(pred_smi)
                        tanimotoa = morganfpTc(truth_smi, pred_smi )
                        if tanimotoa > tanimoto:
                            tanimoto = tanimotoa
                            spred = s_pred
                        print(f"|{tanimotoa}|{s_pred.replace(' ', '')}", end='')
                    print(f"|{s_src}")

                    total += 1
                    if s_truth== spred:
                        exact += 1
                    if tanimoto == 1.0:
                        _100 += 1
                    elif tanimoto >= 0.90:
                        _90 += 1
                    elif tanimoto >= 0.85:
                        _85 += 1
                    elif tanimoto >= 0.80:
                        _80 += 1
                    elif tanimoto >= 0.70:
                        _70 += 1
                    elif tanimoto >= 0.60:
                        _60 += 1
                    elif tanimoto >= 0.50:
                        _50 += 1
                    elif tanimoto >= 0.40:
                        _40 += 1
                    elif tanimoto >= 0.30:
                        _30 += 1
                    elif tanimoto >= 0.20:
                        _20 += 1
                    elif tanimoto > 0.0:
                        _00 += 1
                    else:
                        _invalid += 1

                    tanimoto_similarities.append(tanimoto)

                # ------- model performance evaluation ends  -----
            if args.test:
                break

    end_time = datetime.datetime.now()
    validation_time = end_time - start_time
    seconds = validation_time.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    elapsed_time = f"{hours}hrs {minutes}mins {seconds}secs"

    print(f"{np.mean(tanimoto_similarities) = }")
    print(f"{total, exact, _100, _90, _85, _80, _70, _60, _50, _40, _30, _20, _00, _invalid} \t{elapsed_time=}")


def test_evaluate(args):
    args.root_dir = root_dir
    args.data_dir = data_dir
    args.sp_dir = sp_dir
    args.testset_name =  TEST_DATA
    [ print(f'{i :>15} : {j}') for i,j in vars(args).items()]
    print()

    for fp in fp_names:
        args.fp = fp
        for model_type in ['smiles', 'selfies']:
            args.model_type = model_type
            args.src_vocab_size = fp_vocab_sizes[args.fp]
            args.trg_vocab_size = trg_vocab_sizes[args.model_type]
            args.src_seq_len = fp_seq_lens[args.fp]
            args.trg_seq_len = trg_seq_len
            args.batch_size = 3 #batch_sizes[args.fp]

            args.root_dir = root_dir
            args.fp_datadir = data_dir.joinpath('fingerprints', args.fp)

            args.src_sp_prefix = f"{sp_dir}/{args.fp}_{SP_NAME}"
            args.trg_sp_prefix = f"{sp_dir}/{args.model_type}_{SP_NAME}"

            args.rank  = torch.device('cuda')  if torch.cuda.is_available() else torch.device('cpu')
            args.ddp = False
            device = args.rank

            args.checkpoint = f"{ckpt_dir}/{args.fp}_{args.model_type}_checkpoint.pth"
            args.testset = f'{data_dir}/fingerprints/{args.fp}.{args.model_type}.{TEST_DATA}'
            print('***', args.fp, args.model_type, '***')

            model = build_model(args)

            # Data loading code
            valid_loader = get_data_loader(args.testset, args)
            if args.checkpoint:
                model = setup(model, args.checkpoint, args)
            print()

    #print('Here we go..')
    #[ print(f'{i :>15} : {j}') for i,j in vars(args).items()]
    print(f"Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp',  help='Name of the fingerprint')
    parser.add_argument('--model_type', help='The representation of molecules, either "smiles" or "selfies"')
    parser.add_argument('--test', action='store_true',  help='Test all the checkpoint."')

    parser.add_argument('--testset', '--file', metavar='PATH', help="Read input sequence from a file")
    parser.add_argument('--checkpoint', type=str, help="Checkpoint file for given model.")
    parser.add_argument('--decode', type=str, default='greedy', help="greedy or beam?")
    parser.add_argument('--start_step', default=0, type=int, metavar='N',  help='manual epoch number (useful on restarts)')


    args = parser.parse_args()
    if args.test:
        test_evaluate(args)
    else:
        # checking the settings...
        assert args.fp in fp_names, f"Choose one the fingerprints: \n{fp_names}"
        assert args.model_type in ['smiles', 'selfies'], f"Enter either 'smiles' or 'selfies'"

        if args.testset:
            assert os.path.exists(args.testset), f"There is no testset named {args.testset}."
        else:
            args.testset = f'{data_dir}/fingerprints/{args.fp}.{args.model_type}.{VALID_DATA}'

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
        args.batch_size = 10 #batch_sizes[f'{args.fp}.{args.model_type}']

        args.root_dir = root_dir
        args.fp_datadir = data_dir.joinpath('fingerprints', args.fp)

        args.src_sp_prefix = f"{sp_dir}/{args.fp}_{SP_NAME}"
        args.trg_sp_prefix = f"{sp_dir}/{args.model_type}_{SP_NAME}"

        args.rank  = torch.device('cuda')  if torch.cuda.is_available() else torch.device('cpu')
        args.ddp = False
        device = args.rank

        print('Here we go..')
        [ print(f'{i :>15} : {j}') for i,j in vars(args).items()]
        print(f"")

        model = build_model(args)

        # Data loading code
        valid_loader = get_data_loader(args.testset, args)
        if args.checkpoint:
            model = setup(model, args.checkpoint, args)
            print('PredictionStep|',)
            evaluate(args, model, args.model_type, valid_loader, device=args.rank,  method=args.decode)
            print('PredictionEnd|', )

        else:
            for ckpt in Path(ckpt_dir).iterdir():
                if ckpt.name.startswith('checkpoint_ddp_'):
                    print(ckpt)
                    _step = re.search('checkpoint_ddp_(.*).pth', ckpt.name).group(1)

                    if int(_step) < args.start_step:
                        continue
                    model = setup(model, ckpt, args)

                    print('PredictionStep|', _step)
                    evaluate(args, model, args.model_type, valid_loader, device=args.rank,  method=args.decode)
                    print('PredictionEnd|', _step)

        print('Done!', args.rank)

import os
from pathlib import Path
#import torch

# Path or parameters for data
#root_dir = Path('/home/islambek/projects/fp_benchmark/package')
root_dir = Path('./').resolve()
data_dir = root_dir.joinpath('data')
sp_dir = root_dir.joinpath('data', 'sp')

TRAIN_DATA = 'test'
VALID_DATA = 'test'
TEST_DATA = 'test'
SP_NAME = 'vocab_sp'

fp_names = ['MACCS', 'Avalon', 'RDK4', 'RDK4-L', 'HashAP', 'TT', 'HashTT', 'ECFP0', 'ECFP2', 'ECFP4', 'FCFP2', 'FCFP4', 'AEs' ]

fp_vocab_sizes = \
        {"MACCS": 160,#
         "Avalon": 516, #1028,#
         "RDK4": 2052, #
         "RDK4-L": 2052, #
         "HashAP": 1992, # 1998, #1958,
         "TT": 54973, #65880, #34212,
         "HashTT": 2052, #
         "ECFP0": 98, #100, #99,
         "ECFP2": 2052, #
         "ECFP4": 2052, #
         "FCFP2": 1492, #1576, #1276,
         "FCFP4": 2052,
         "AEs" :54076, #67214 ,#65407, #30842
         }

trg_seq_len = 130
fp_seq_lens =\
        {"MACCS": 107, #avg50 #105,
         "Avalon": 470, # avg200 #717,
         "RDK4": 290, #avg85 #276,
         "RDK4-L": 210, #avg60 #209,
         "HashAP": 275, #avg100 #266,
         "TT": 125, #avg35 #111,
         "HashTT": 118, #avg35
         "ECFP0": 25, #avg10 #25,
         "ECFP2": 65, #avg30 #63,
         "ECFP4": 104, #avg50 #100,
         "FCFP2": 51, #avg20 #50,
         "FCFP4": 86, #avg 36 #86
         "AEs" : 66, #avg30, #65
         }

batch_sizes =\
        {'MACCS.smiles': 163,
         'MACCS.selfies': 174,
         'Avalon.smiles': 69,
         'Avalon.selfies': 71,
         'RDK4.smiles': 120,
         'RDK4.selfies': 126,
         'RDK4-L.smiles': 142,
         'RDK4-L.selfies': 157,
         'HashAP.smiles': 113,
         'HashAP.selfies': 118,
         'TT.smiles': 198,
         'TT.selfies': 214,
         'HashTT.smiles': 198,
         'HashTT.selfies': 214,
         'ECFP0.smiles': 267,
         'ECFP0.selfies': 296,
         'ECFP2.smiles': 205,
         'ECFP2.selfies': 222,
         'ECFP4.smiles': 167,
         'ECFP4.selfies': 178,
         'FCFP2.smiles': 232,
         'FCFP2.selfies': 254,
         'FCFP4.smiles': 188,
         'FCFP4.selfies': 203,
         'AEs.smiles': 205,
         'AEs.selfies': 222}

trg_vocab_sizes = { 'smiles':109, #avg51
                   'selfies': 205, #avg44
                  }

# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
character_coverage = 1.0
sp_model_type = 'word'

# Parameters for Transformer & training
num_heads = 8
num_layers = 6
dim_model = 512
dim_ff = 2048
dim_k = dim_model // num_heads
dropout_rate = 0.1

train_step = 500000
beam_size = 10
learning_rate = 0.001

ckpt_dir = 'saved_models'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

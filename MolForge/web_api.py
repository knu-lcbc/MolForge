from .parameters import *
from .predict import setup, greedy_search, beam_search
from .utils import *

import torch


def model_call(input, fp='ECFP4', model_type='smiles', checkpoint=None, decode='greedy'):

    class Args:
        pass

    args = Args
    args.fp = fp
    args.model_type = model_type
    args.checkpoint = checkpoint
    args.decode = decode

    assert args.fp in fp_names, f"Choose one the fingerprints: \n{fp_names}"
    assert args.model_type in ['smiles', 'selfies'], f"Enter either 'smiles' or 'selfies'"

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

    model = setup(build_model(args, print_=False).to(args.device), args.checkpoint, args, False)

    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{args.src_sp_prefix}.model")
    trg_sp.Load(f"{args.trg_sp_prefix}.model")

    #print("Preprocessing input sequence...")
    tokenized = src_sp.EncodeAsIds(input)
    src_data = torch.LongTensor(pad_or_truncate(tokenized, args.src_seq_len)).unsqueeze(0).to(args.device) # (1, L)
    e_mask = (src_data != pad_id).unsqueeze(1).to(args.device) # (1, 1, L)

    #start_time = datetime.datetime.now()

    #print("Encoding input sequence...")
    model.eval()
    src_data = model.src_embedding(src_data)
    src_data = model.src_positional_encoder(src_data)
    e_output = model.encoder(src_data, e_mask) # (1, L, d_model)

    if decode == 'greedy':
        #print("Greedy decoding selected.")
        result = greedy_search(model, e_output, e_mask, trg_sp, args.device, False)
    elif decode == 'beam':
        #print("Beam search selected.")
        result = beam_search(model, e_output, e_mask, trg_sp, args.device, False)
    #print()

    #end_time = datetime.datetime.now()

    #total_inference_time = end_time - start_time
    #seconds = total_inference_time.seconds
    #minutes = seconds // 60
    #seconds = seconds % 60

    #print(f"Input: {input}")
    #print(f"Result: {result}")
    #print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")

    return result

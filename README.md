[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Reconstruction of lossless molecular representations.
SMILES is the most dominant molecular representation used in AI-based chemical applications, but it has innate limitations associated with its internal structure.
Here, we exploit the idea that a set of structural fingerprints can be used as efficient alternatives to unique molecular representations.
For this purpose, we trained neural-machine-translation based models that translate a set of various structural fingerprints to conventional text-based molecular representations, i.e., SMILES and SELFIES. 
The assessment of their conversion efficiency showed that our models successfully reconstructed molecules and achieved a high level of accuracy. 
Therefore, our approach brings structural fingerprints into play as strong representational tools in chemical natural language processing applications by restoring the connectivity information that is lost during fingerprint transformation.
This comprehensive study addressed the major limitation of structural fingerprints, which precludes their implementation in NLP models.
Our findings would facilitate the development of text or fingerprint-based chemoinformatic models for generative and translational tasks.

<hr style="background: transparent; border: 0.2px dashed;"/>

### Code usage

#### Requirements
The source code is tested on Linux operating systems. After cloning the repository, we recommend creating a new conda environment and install the package locally. Users should install required packages described in `environments.yml` prior to direct use.

   ```shell
   conda env create --name MolForge_env --file=environment.yml
   conda activate MolForge_env
   pip install .
   ```
   
#### Prediction & Demo:

First, checkpoint files ([top-performing](https://drive.google.com/uc?id=1zl6HBdwYsnA4JcnOi1o6OmcrRDB5iySK) or [all the oher models](https://drive.google.com/uc?id=1jCtbc9lMacCyiZ3iZFEtFgOfOQYtWEuD)) should be downloaded and extracted. The checkpoints files should be placed in `./saved_models/` directory. Then,run below commands to conduct an inference with the trained model.

   ```shell
   python predict.py --fp  --model_type --input --checkpoint
   ```
   where: 
   - `--fp` : The name of fingerprint.
   - `--model_type` : Molecular representation e.g. 'smiles' or 'selfies'
   - `--input` : Bit number of the fingerprint (`--fp`).
   - `--checkpoint` : Checkpoint file for the given model. If `None`, it uses the downloaded checkpoints in the `./saved_models/`.  
   - `--decode`: Decoding algorithm (either `'greedy'` or `'beam'`), (by default: `greedy`)
 

Example prediction;
```shell
predict --fp='ECFP4' --model_type='smiles' --input='1 80 94 114 237 241 255 294 392 411 425 695 743 747 786 875 1057 1171 1238 1365 1380 1452 1544 1750 1773 1853 1873 1970'
```
and its sample output;

   ```shell
Here we go..
             fp : ECFP4
     model_type : smiles
          input : 1 80 94 114 237 241 255 294 392 411 425 695 743 747 786 875 1057 1171 1238 1365 1380 1452 1544 1750 1773 1853 1873 1970
     input_file : None
     checkpoint : saved_models/ECFP4_smiles_checkpoint.pth
         decode : greedy
 src_vocab_size : 2052
 trg_vocab_size : 109
    src_seq_len : 104
    trg_seq_len : 130
       root_dir : /home/tmp/MolForge
     fp_datadir : /home/tmp/MolForge/data/fingerprints/ECFP4
  src_sp_prefix : /home/tmp/MolForge/data/sp/ECFP4_vocab_sp
  trg_sp_prefix : /home/tmp/MolForge/data/sp/smiles_vocab_sp
           rank : cuda
         device : cuda

The size of src vocab is 2052 and that of trg vocab is 109.
Loading checkpoint... ECFP4 smiles
Preprocessing input sentence...
Encoding input sentence...
Greedy decoding selected.

Input: 1 80 94 114 237 241 255 294 392 411 425 695 743 747 786 875 1057 1171 1238 1365 1380 1452 1544 1750 1773 1853 1873 1970
Result: C C O C 1 = C ( C = C ( C = C 1 ) C ( C ( C ) ( C ) C ) N ) O C C
Inference finished! || Total inference time: 0mins 0secs

   ```

   
<hr style="background: transparent; border: 0.5px dashed;"/>

## Result
Each cell shows the Tanimoto exactness (%) of selected fingerprint transformation to SMILES (row-wise) computed at the respective fingerprint encodings(columns-wise). The consistency in color code reflects the robustness, while the jumps represent the effect of selection bias. ECFP2* and ECFP4* represent explicit bit versions.

|        |   MACCS |   Avalon |   RDK4 |   RDK4_L |   HashAP |   TT |   HashTT |   ECFP0 |   ECFP2 |   ECFP4 |   FCFP2 |   FCFP4 |   AEs |   ECFP2* |   ECFP4* |
|:-------|--------:|---------:|-------:|-----------:|---------:|-----:|---------:|--------:|--------:|--------:|--------:|--------:|------:|---------:|---------:|
| MACCS  |    77.4 |     33.3 |   38   |     39.8 |     32.2 | 33.2 |     33.2 |    52.2 |    34.7 |    32.5 |    48.6 |    33.5 |  34.7 |     37   |     33.3 |
| Avalon |    72.6 |     67.9 |   72.2 |     73.5 |     63.4 | 64.7 |     64.7 |    69.5 |    65.6 |    63.6 |    68.9 |    64.7 |  65.6 |     68.5 |     64.6 |
| RDK4   |    66.9 |     60   |   90.9 |     91.5 |     59.8 | 61.1 |     61.1 |    62.5 |    60.2 |    58.3 |    62.3 |    59.6 |  60.2 |     64.3 |     59.6 |
| RDK4_L |    52.6 |     46.7 |   64.7 |     88.8 |     46.7 | 47.7 |     47.7 |    49.1 |    46.9 |    45.5 |    48.8 |    46.5 |  46.9 |     49.3 |     46.2 |
| HashAP |    86.5 |     83.8 |   89.6 |     90.2 |     85.2 | 85.5 |     85.5 |    84.3 |    83.1 |    82.5 |    84   |    82.8 |  83.1 |     86.1 |     84.1 |
| TT     |    88.4 |     83.5 |   92.3 |     92.5 |     84.1 | 87.3 |     87.3 |    85.8 |    85.2 |    82.3 |    85.7 |    83.8 |  85.2 |     91.4 |     84.2 |
| HashTT |    86.2 |     81.4 |   90.2 |     90.5 |     82.1 | 85.3 |     85.5 |    83.9 |    83.3 |    80.4 |    83.8 |    81.8 |  83.3 |     89.2 |     82.2 |
| ECFP0  |     3.3 |      1.3 |    2.1 |      2.7 |      1.2 |  1.3 |      1.3 |     4   |     1.4 |     1.2 |     2.9 |     1.3 |   1.4 |      1.8 |      1.4 |
| ECFP2  |    86   |     75.8 |   83.1 |     83.1 |     73.6 | 76   |     76   |    84.7 |    82.7 |    74.4 |    84.5 |    76.5 |  82.7 |     96.2 |     76   |
| ECFP4  |    95.1 |     92.6 |   95.7 |     95.7 |     90.8 | 92.4 |     92.4 |    93.5 |    93.1 |    92.1 |    93.3 |    92.4 |  93.1 |     96.6 |     94.8 |
| FCFP2  |    25.6 |     16.3 |   20.1 |     21.6 |     15.5 | 16   |     16   |    28.6 |    16.9 |    15.7 |    38.7 |    20.4 |  16.9 |     19.6 |     16.1 |
| FCFP4  |    71.5 |     67.5 |   73.7 |     73.8 |     65.5 | 67.3 |     67.3 |    69.2 |    68.5 |    66.3 |    87.6 |    86.7 |  68.5 |     74.4 |     68.1 |
| AEs    |    86.7 |     76.2 |   83.5 |     83.6 |     74   | 76.3 |     76.3 |    85.3 |    83.5 |    74.7 |    85.2 |    76.8 |  83.5 |     97   |     76.5 |

For more results see the `Main_Results.ipynb` notebook.
<hr style="background: transparent; border: 0.5px dashed;"/>

## Cite
[![DOI](https://zenodo.org/badge/451459811.svg)](https://zenodo.org/badge/latestdoi/451459811)

### License

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

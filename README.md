# Towards Effectively Testing Machine Translation Systems from White-Box Perspectives

This repo contains the data and the source code of  the tool (i.e., **GRI** and **WALI**) used in our paper *Towards Effectively Testing Machine Translation Systems from White-Box Perspectives*. 

The `CAT` folder contains all source files and data used to reproduce the results of `CAT` approach. The `GRI` contains all source files and data used to reproduce the results of `GRI` approach. The `WALI` contains all source files and data used to reproduce the results of `WALI` approach. The `Labeled data` folder contains the samples of data used in our paper for manual evaluation.

The Transformer model used for the implementation and evaluation can be found at [transformer model](https://mega.nz/file/tDlyiSbJ#uz36-pUyrM6qXnj2h97BkKjOp4otVVTevqKi4axkpH8)

## Requirements and Installation
```bash
git clone https://github.com/conf2024-8888/NMT-Testing.git
cd NMT-Testing
pip install -r requirements.txt
```
## Replicate the results
> 1. download transformer model at [transformer model](https://mega.nz/file/tDlyiSbJ#uz36-pUyrM6qXnj2h97BkKjOp4otVVTevqKi4axkpH8) and unzip the zipped file and put the model in the root directory of the repository.

> 2. Go to the subdirectory **CAT**, **GRI** or **WALI**: 

```bash
cd GRI
```

> 3. Run the pipeline.sh : 

```bash 
sh pipeline.sh
```

> 4. The bash file **pipeline.sh** inside each subfolder will trigger all source code to run.


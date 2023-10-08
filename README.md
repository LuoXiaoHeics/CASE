# ECASE

## Requirements

The original project is tested under the following environments:

```
python==3.8.2
torch==1.8.1+cu111
pytorch-lightning==1.5.0
transformers==4.10.0
scikit-learn==1.2.1
```

## Data

Please download the .jsonl files and store them under `./data`.

The datasets can be downloaded using the links below (requires format conversion, code can be found in [`./scripts/`](https://github.com/bloomberg/argument-relation-transformer-acl2022):

- Essays (Stab and Gurevych, 2017): [link](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422)
- AbstRCT (Mayer et al., 2020): [link](https://gitlab.com/tomaye/abstrct/)
- ECHR (Poudyal et al., 2020): [link](http://www.di.uevora.pt/~pq/echr/)
- CDCP (Park and Cardie, 2018; Niculae et al., 2017): [link](https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip)
- AMPERE++ [link](https://zenodo.org/record/6362430#.YjJJUprMIba)

Or you can directly download the processed data from [link](https://drive.google.com/file/d/1FKyZtl1TFfwwZ0n_RaBTs7LSvxqvOZmk/view?usp=sharing).

## Quick Start
To train a standard supervised relation extraction model on CDCP:

```shell script
SEED=1
DOMAIN="cdcp"

python -m argument_relation_transformer.train \
  --datadir=./data \
  --seed=${SEED} \
  --dataset=${DOMAIN} \
  --ckptdir=./checkpoints \
  --exp-name=demo-${DOMAIN}_seed=${SEED} \
  --warmup-steps=5000 \
  --learning-rate=1e-5 \
  --scheduler=constant \
  --max-epochs=15 \
  --window-size=10 \
  --cl=1e-3
```


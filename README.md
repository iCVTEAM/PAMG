# PAMG

This repository contains the PyTorch implementation of **PAMG**, a text-to-motion generation pipeline for part-aware motion generation. PAMG builds on the MoMask-style masked-and-residual generation framework, replaces the text-conditioning branch with a frozen `BAAI/bge-large-en-v1.5` encoder, and represents motion with separate RVQ tokenizers for body parts.

[Pretrained Models](MODEL_ZOO.md) | [Data Preparation](DATA.md)


## Release Plans

- [x] Part-aware RVQ tokenizer training code
- [x] BGE-conditioned masked transformer training code
- [x] BGE-conditioned residual transformer training code
- [x] HumanML3D and KIT-ML data interfaces
- [x] Evaluation code for the BGE text-to-motion pipeline
- [ ] Clean single-prompt BGE generation script
- [ ] Public checkpoint links and checksums
- [ ] Final paper/project links

## Contents

- `train_vq_con.py`: trains the part-aware RVQ tokenizer used by PAMG.
- `train_t2m_transformer_bge.py`: trains the BGE-conditioned masked transformer.
- `train_res_transformer_bge.py`: trains the BGE-conditioned residual transformer.
- `eval_t2m_trans_res_bge.py`: evaluates text-to-motion generation with the BGE masked and residual transformers.
- `models/mask_transformer_bge/`: BGE-conditioned transformer and trainer modules.
- `models/vq_seg_con/`: part-aware RVQ-VAE implementation used by the released PAMG path.
- `data/`, `motion_loaders/`, `utils/`: HumanML3D/KIT-ML loading, evaluation, and plotting utilities inherited from MoMask-style pipelines.
- `prepare/`: helper scripts for public evaluator assets and GloVe files once links are finalized.

The public release is centered on the PAMG/BGE path above; experimental branches from the research workspace are intentionally kept out of this tree.

## Installation

We recommend creating a fresh environment before running the code.

```bash
conda create -n pamg python=3.10
conda activate pamg
pip install -r requirements.txt
```

PAMG uses Hugging Face Transformers and downloads `BAAI/bge-large-en-v1.5` on first use. If you run on a cluster or an offline machine, download/cache the model beforehand and make sure the Hugging Face cache is visible to the job.

```bash
python - <<'PY'
from transformers import AutoModel, AutoTokenizer
name = "BAAI/bge-large-en-v1.5"
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print("BGE model is available:", name)
PY
```

Optional rendering or BVH export scripts are not part of this minimal release. SMPL/SMPL-X assets and other third-party visualization resources are not redistributed in this repository.

## Data Preparation

This release does not include HumanML3D, KIT-ML, GloVe, evaluator checkpoints, or generated caches. Please prepare them locally following the licenses of the original datasets.

Expected layout:

```text
dataset/
  HumanML3D/
    new_joint_vecs/
    texts/
    Mean.npy
    Std.npy
    train.txt
    val.txt
    test.txt
  KIT-ML/
    new_joint_vecs/
    texts/
    Mean.npy
    Std.npy
    train.txt
    val.txt
    test.txt
checkpoints/
  t2m/
    Comp_v6_KLD005/
  kit/
    Comp_v6_KLD005/
```

See [DATA.md](DATA.md) for the full data and evaluator layout.

## Pretrained Models

Pretrained weights are not included in git. Place downloaded checkpoints under:

```text
checkpoints/<dataset_name>/<experiment_name>/
  opt.txt
  meta/
    mean.npy
    std.npy
  model/
    net_best_fid.tar
    latest.tar
```

PAMG expects three model families:

```text
checkpoints/t2m/<part_rvq_name>/
checkpoints/t2m/<bge_masked_transformer_name>/
checkpoints/t2m/<bge_residual_transformer_name>/
```

See [MODEL_ZOO.md](MODEL_ZOO.md) for the checkpoint table and naming placeholders.

## Training

Commands below assume they are run from the repository root. Replace the experiment names with your own names if you use different checkpoints.

### 1. Train the part-aware RVQ tokenizer

```bash
python train_vq_con.py \
  --name rvq_con_wogq0.2nb512 \
  --dataset_name t2m \
  --gpu_id 0 \
  --batch_size 512 \
  --nb_code 512 \
  --num_quantizers 6 \
  --max_epoch 50 \
  --quantize_dropout_prob 0.2
```

### 2. Train the BGE masked transformer

```bash
python train_t2m_transformer_bge.py \
  --name mtrans_bge_b64n6dp0.1wog \
  --dataset_name t2m \
  --gpu_id 0 \
  --batch_size 64 \
  --vq_name rvq_con_wogq0.2nb512 \
  --n_layers 6 \
  --max_epoch 500
```

### 3. Train the BGE residual transformer

```bash
python train_res_transformer_bge.py \
  --name rtrans_bge_b64dp0.2wog \
  --dataset_name t2m \
  --gpu_id 0 \
  --batch_size 64 \
  --vq_name rvq_con_wogq0.2nb512 \
  --cond_drop_prob 0.2 \
  --share_weight
```

Training outputs are written to:

```text
checkpoints/<dataset_name>/<experiment_name>/
log/<stage>/<dataset_name>/<experiment_name>/
```

## Evaluation

Evaluate the BGE masked transformer together with the BGE residual transformer:

```bash
python eval_t2m_trans_res_bge.py \
  --dataset_name t2m \
  --name mtrans_bge_b64n6dp0.1wog \
  --res_name rtrans_bge_b64dp0.2wog \
  --gpu_id 0 \
  --cond_scale 4 \
  --time_steps 10 \
  --ext evaluation
```

Evaluation logs are saved to:

```text
checkpoints/<dataset_name>/<masked_transformer_name>/eval/<ext>.log
```

The script reports FID, Diversity, R-Precision, Matching Score, and Multimodality following the MoMask/Text2Motion evaluation protocol.

## Acknowledgements

This codebase is built on the MoMask text-to-motion framework and its related HumanML3D/KIT-ML evaluation pipeline. We thank the authors of MoMask, HumanML3D, T2M-GPT, MDM, MLD, vector-quantize-pytorch, Hugging Face Transformers, and BAAI/FlagEmbedding for their open-source work.

## License

The code is released under the MIT license unless otherwise noted. Datasets, evaluator checkpoints, SMPL/SMPL-X assets, BGE model weights, and other third-party assets have their own licenses and must be obtained separately by users.

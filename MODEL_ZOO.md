# Model Zoo

Pretrained checkpoints are not included in git. Public links and checksums will be added after final hosting and license review.

## PAMG Checkpoints

| Name | Dataset | Description | Link | SHA256 | Status |
| --- | --- | --- | --- | --- | --- |
| `rvq_con_wogq0.2nb512` | HumanML3D | Part-aware RVQ tokenizer | TBD | TBD | Pending |
| `mtrans_bge_b64n6dp0.1wog` | HumanML3D | BGE-conditioned masked transformer | TBD | TBD | Pending |
| `rtrans_bge_b64dp0.2wog` | HumanML3D | BGE-conditioned residual transformer | TBD | TBD | Pending |
| `rvq_con_kit` | KIT-ML | Part-aware RVQ tokenizer | TBD | TBD | Pending |
| `mtrans_bge_kit` | KIT-ML | BGE-conditioned masked transformer | TBD | TBD | Pending |
| `rtrans_bge_kit` | KIT-ML | BGE-conditioned residual transformer | TBD | TBD | Pending |

## Expected Directory Layout

Place downloaded checkpoints under `checkpoints/<dataset>/<experiment>/`:

```text
checkpoints/t2m/rvq_con_wogq0.2nb512/
  opt.txt
  meta/
    mean.npy
    std.npy
  model/
    net_best_fid.tar
    latest.tar

checkpoints/t2m/mtrans_bge_b64n6dp0.1wog/
  opt.txt
  model/
    net_best_fid.tar
    latest.tar

checkpoints/t2m/rtrans_bge_b64dp0.2wog/
  opt.txt
  model/
    net_best_fid.tar
    latest.tar
```

## BGE Text Encoder

The released BGE path uses:

```text
BAAI/bge-large-en-v1.5
```

The model is loaded through Hugging Face Transformers. It is not committed to this repository. Users should download it through Hugging Face or an approved internal cache according to the model license.

## Files Kept Out Of Git

```text
checkpoints/
generation/
log/
*.pt
*.pth
*.tar
*.ckpt
*.bin
*.safetensors
*.npy
*.npz
*.pkl
*.mp4
*.gif
```

## Release TODO

- Upload public checkpoints.
- Add download URLs.
- Add SHA256 checksums.
- Confirm license/usage terms for every checkpoint.
- Add a minimal inference example once `gen_t2m_bge.py` is finalized.

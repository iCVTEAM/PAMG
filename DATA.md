# Data Preparation

This repository does not redistribute HumanML3D, KIT-ML, evaluator checkpoints, GloVe files, or generated caches. Please obtain each dataset from its official source and follow its license.

## HumanML3D

Prepare HumanML3D with the standard text-to-motion preprocessing pipeline. The expected layout is:

```text
dataset/HumanML3D/
  new_joint_vecs/
  texts/
  Mean.npy
  Std.npy
  train.txt
  val.txt
  test.txt
  train_val.txt
  all.txt
```

The PAMG code reads motion features from `new_joint_vecs/` and text annotations from `texts/`.

## KIT-ML

Prepare KIT-ML in the same format:

```text
dataset/KIT-ML/
  new_joint_vecs/
  texts/
  Mean.npy
  Std.npy
  train.txt
  val.txt
  test.txt
  train_val.txt
  all.txt
```

## Evaluator Assets

Training and evaluation use the same evaluator layout as MoMask/Text2Motion:

```text
checkpoints/
  t2m/
    Comp_v6_KLD005/
      opt.txt
      model/
  kit/
    Comp_v6_KLD005/
      opt.txt
      model/
glove/
  our_vab_data.npy
  our_vab_idx.pkl
  our_vab_words.pkl
  glove_embedding.npy
```

The helper scripts in `prepare/` can be used once the public links are finalized:

```bash
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```

## Notes

- Do not commit raw datasets or processed dataset arrays.
- Do not commit GloVe files or evaluator checkpoints.
- Keep local dataset paths relative to the repository root, as shown above.
- For a clean clone test, verify that `python train_t2m_transformer_bge.py --help` works before adding data and checkpoints.

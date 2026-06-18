# Notice

This repository releases PAMG, a part-aware text-to-motion generation codebase built on a MoMask-style backbone with BGE text conditioning.

## Upstream Code

The implementation builds on MoMask and related text-to-motion utilities. Keep the upstream MIT license and attribution when redistributing this code.

## BGE / FlagEmbedding

The BGE text encoder used by PAMG is `BAAI/bge-large-en-v1.5`, loaded through Hugging Face Transformers. The BGE model weights are not redistributed in this repository. Users must obtain the model from its upstream provider and follow its license.

## Datasets And Assets

HumanML3D, KIT-ML, evaluator checkpoints, GloVe files, SMPL/SMPL-X assets, pretrained model weights, generated videos, logs, and local experiment outputs are not included in git. They may have separate licenses or redistribution terms.

## Disclaimer

This notice is provided for engineering release clarity and is not legal advice. Confirm final license compatibility before public publication.

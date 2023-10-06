# ZeroGen: Zero-shot Multimodal Controllable Text Generation with Multiple Oracles
**[In Progress]** Official PyTorch implementation of ZeroGen: Zero-shot Multimodal Controllable Text Generation with Multiple Oracles (https://arxiv.org/abs/2306.16649), accepted to NLPCC 2023.

![teaser](./teaser.jpg)

## Setup

Make sure you have installed:
```bash
transformers
nltk
scikit-learn
torch
numpy
tqdm
```

## Data and Model Weights

### Data

The [extra data](https://drive.google.com/drive/folders/1XHviYZnrX3KNqSKvUwkoHsxmeSFP5Jgn?usp=sharing) contains:

1. Objects, textual features, ect. for `MSCOCO`, `Flickr30k`, `Flickr10k`.
2. The training data for `Flickr10k`.
3. `evaluation` suite for captioning and text control evaluations.

Put the datasets to your directory and change the `config.json` file accordingly, and put the `evaluation` folder to the current directory.

For the test data of `MSCOCO` and `Flickr30k`, please refer to the downloading details from [this repository](https://github.com/yxuansu/MAGIC).

### Model Weights

| Task               | Weight                                                   |
| :----------------- | :------------------------------------------------------- |
| MSCOCO             | https://huggingface.co/cambridgeltl/magic\_mscoco        |
| Flickr30k          | https://huggingface.co/cambridgeltl/magic\_flickr30k     |
| Flickr10k-romantic | https://huggingface.co/PahaII/ZeroGen-flickr10k-romantic |
| Flickr10k-humor    | https://huggingface.co/PahaII/ZeroGen-flickr10k-humor    |
| VisNews            | https://huggingface.co/PahaII/ZeroGen-visnews            |

## Inference

```bash
TASK=mscoco
LENGTH=16
ALPHA=1.0
BETA=1.0
ETA=0.10
K=45
ALPHA_HAT=2.5
BETA_HAT=1.0
N=1

python run_zerogen.py --alpha ${ALPHA} --beta ${BETA} --eta ${ETA} --k ${K} --condition_method add \
                       --task ${TASK} --decoding_len ${LENGTH} --alpha_scale --alpha_activasize ${ALPHA_HAT}  \
                       --beta_scale --beta_activesize 0.2 --beta_upper ${BETA_HAT} --n_obj ${N}
```

Here are recommended parameters for ZeroGen generation:

| Task               | $k$ | $\alpha$ | $\beta$ | $\eta$ | $\hat{\alpha}$ | $\hat{\beta}$ | $N$ | length
| :----------------- | :---- | :---------- | :--------- | :-------- | :----------------- | :---------------- | :---- | :---- |
| MSCOCO             | 45    | 1\.0        | 1\.0       | 0\.10     | 2\.5               | 1\.0              | 1~5   | 16
| Flickr30k          | 25    | 2\.0        | 1\.0       | 0\.10     | 2\.0               | 0\.5              | 1~5   | 16
| Flickr10k-romantic | 45    | 1\.0        | 1\.0       | 0\.10     | 3\.0               | 0\.5              | 1     | 25
| Flickr10k-humor    | 45    | 1\.0        | 1\.0       | 0\.10     | 2\.5               | 0\.5              | 1     | 25
| VisNews            | 5     | 8\.0        | 1\.0       | 0\.65     | 8\.0               | 0\.5              | 40    | 64

We also support the inference of sequence-to-sequence models like [FlanT5](https://huggingface.co/google/flan-t5-base), just add `--seq2seq` flag and specify the model name via `--language_model_name` argument.

## Citation

If you find our work useful, please consider cite our paper and star the repo

```bibtex
@article{tu2023zerogen,
  title={ZeroGen: Zero-shot Multimodal Controllable Text Generation with Multiple Oracles},
  author={Tu, Haoqin and Yang, Bowen and Zhao, Xianfeng},
  journal={arXiv preprint arXiv:2306.16649},
  year={2023}
}
```
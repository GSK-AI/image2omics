# Multi-omics Prediction from High-content Cellular Imaging with Deep Learning
Rahil Mehrizi, Arash Mehrjou, Maryana Alegro, Yi Zhao, Benedetta Carbone, Carl Fishwick, Johanna Vappiani, Jing Bi, Siobhan Sanford, Hakan Keles, Marcus Bantscheff, Cuong Nguyen, Patrick Schwab

[[`GSK AI`](https://www.gsk.ai/)] [[`Paper`](https://arxiv.org/abs/2306.09391)] [[`Blog`]()] [[`BibTeX`](#citation)]

Accompanying code for Image2Omics. For details, see **Multi-omics Prediction from High-content Cellular Imaging with Deep Learning**.

## Instructions
#### Cloning and setting up your environment
```bash
git clone https://github.com/GSK-AI/image2omics
cd image2omics
conda env create --name image2omics --file environment.yaml
source activate image2omics
source .env
```
#### Downloading checkpoints and data
```bash
export DATA_DIR=image2omics-data
aws s3 sync s3://image2omics $DATA_DIR --no-sign-request
```

## Use
#### Reproducing results in manuscript
```bash
source run_pipeline.sh $DATA_DIR $DATA_DIR/results false
```
#### Rerunning inference from pretrained checkpoints
```bash
source run_pipeline.sh $DATA_DIR $DATA_DIR/results true
```

## License
Image2Omics code is released under the [Apache 2.0 license](LICENSE).

## Citation
Please consider citing, if you reference or use our methodology, code or results in your work:
```
@article{mehrizi2023multi,
  title={Multi-omics Prediction from High-content Cellular Imaging with Deep Learning},
  author={Mehrizi, Rahil and Mehrjou, Arash and Alegro, Maryana and Zhao, Yi and Carbone, Benedetta and Fishwick, Carl and Vappiani, Johanna and Bi, Jing and Sanford, Siobhan and Keles, Hakan and others},
  journal={arXiv preprint arXiv:2306.09391},
  year={2023}
}
```

# 	ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark [arXiv preprint]

**Authors**: _Joanne Lin, Ruirui Lin, Yini Li, David Bull, Nantheera Anantrasirichai_

**Institution**: Visual Information Laboratory, University of Bristol, United Kingdom

[[`arXiv`](#)]

## Setup
### Dependencies
Create and run conda environment:
```
conda env create -f environment.yml
conda activate vdpnet
```

## Train
Run the following command to train our model:
```
python train.py
```

## Inference
Currently not implemented yet. Feel free to use our released [weights](https://github.com/JoanneLin168/degradation-estimation-network/releases/download/v1.0/best.pt).

## Citation
If you use our work in your research, please cite using the following BibTeX entry:
```
@article{lin2025den,
         title={Towards a General-Purpose Zero-Shot Synthetic Low-Light Image and Video Pipeline},
         author={Lin, Joanne and Morris, Crispian, and Lin, Ruirui and Zhang, Fan and Bull, David and Anatrasirichai, Nantheera},
         year={2025},
         publisher={arXiv}}
```

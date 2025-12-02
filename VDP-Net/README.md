# 	VDP-Net

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
Run the following command to train our model:
```
python inference.py --input <video> --reference <video> --checkpoint <weights>
```
Feel free to use our released [weights](https://github.com/JoanneLin168/ELVIS/releases/download/v1.0/vdpnet.pth).

# GenVIS + ELVIS

## Setup
Please refer to the original setup [here](https://github.com/miranheo/GenVIS)

## Train
Run the following command to train GenVIS with ELVIS:
```
python train_net_elvis.py --num-gpus 4 --config configs/genvis/lowlight_ytvis19/genvis_R50_bs8_online_elvis.yaml
```

## Inference
Run the following command to run inference using GenVIS + ELVIS:
```
python train_net_elvis.py --num-gpus 4 --eval-only --config configs/genvis/lowlight_ytvis19/genvis_R50_bs8_online_elvis.yaml
```
Feel free to use our released [weights](https://github.com/JoanneLin168/ELVIS/releases/download/v1.0/elvis.pth).

### Evaluation on real low-light dataset
To run evaluation on real low-light dataset, you will need to register your own detectron2 dataset and add the following command:
```
python train_net_elvis.py --num-gpus 4 --eval-only --config configs/genvis/lowlight_ytvis19/genvis_R50_bs8_online_elvis.yaml DATASETS.TEST <dataset>
```

> NOTE: remember to change the dataset mapper in the `build_test_loader()` function in `train_net_elvis.py`.
# SNO_pytorch
Shearlet Neural Operator

## Step 1: generate sheared kelvin helmotz data (KH) 
Following **SHEARLET NEURAL OPERATORS FOR ANISOTROPIC-SHOCK-DOMINATED AND MULTI-SCALE PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS**  [paper](https://arxiv.org/pdf/2604.25181)
```bash
# run data generation code
python dataset.py --data_dir data --datasets kh --H 128 --W 128 --n_train 200 --n_test 50
```
## Step 2: train different models on KH dataset
```bash
# run model training
python train_all.py --dataset kh --methods fno sno usno cascade --epochs 200
# run model testing
python test.py --data_dir data --ckpt_dir checkpoints --fig_dir figures --dataset kh --H 128 --W 128 --hidden 32
```

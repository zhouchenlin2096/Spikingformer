# Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network, [Arxiv 2023](https://arxiv.org/abs/2304.11954)
Spikingformer is a pure event-driven transformer-based spiking neural network (75.85% top-1 accuracy on ImageNet, + 1.04% and significantly reduces energy consumption by 57.34% compared with Spikformer).


## Reference
If you find this repo useful, please consider citing:
```
@article{zhou2023spikingformer,
  title={Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network},
  author={Zhou, Chenlin and Yu, Liutao and Zhou, Zhaokun and Zhang, Han and Ma, Zhengyu and Zhou, Huihui and Tian, Yonghong},
  journal={arXiv preprint arXiv:2304.11954},
  year={2023},
  url={https://arxiv.org/abs/2304.11954}
}
```
Our codes are based on the official imagenet example by PyTorch, pytorch-image-models by Ross Wightman and SpikingJelly by Wei Fang.

## Main results on ImageNet-1K

| Model               | Resolution| T |  Param.     | FLOPs   |  Energy Consumption |Top-1 Acc|
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:              |:---: |
| Spikingformer-8-384 | 224x224   | 4 |  16.81M     | 6.82G   | 4.69 mJ   |72.45  |
| Spikingformer-8-512 | 224x224   | 4 |  29.68M     | 11.09G  | 7.46 mJ   |74.79  |
| Spikingformer-8-768 | 224x224   | 4  |  66.34M     | 22.09G  | 13.68 mJ  |75.85  |

<!-- 
| Spikformer-8-384 | 224x224    |  16.81M     | 6.82G   | 12.43  mJ              |70.24  |
| Spikformer-8-512 | 224x224    |  29.68M     | 11.09G  | 18.82  mJ             |73.38  |
| Spikformer-8-768 | 224x224    |  66.34M     | 22.09G  | 32.07  mJ             |74.81  |
-->

## Main results on CIFAR10/CIFAR100

| Model                | T      |  Param.     | CIFAR10 Top-1 Acc |CIFAR100 Top-1 Acc|
| :---:                | :---:  | :---:       |  :---:    |:---: |
| Spikingformer-4-256  | 4      |  4.15M     | 94.77     |77.43  |
| Spikingformer-2-384  | 4      |  5.76M     | 95.22     |78.34  |
| Spikingformer-4-384  | 4      |  9.32M     | 95.61     |79.09  |
| Spikingformer-4-384-400E  | 4      |  9.32M     | 95.81     |79.21  |

## Main results on CIFAR10-DVS/DVS128

| Model               | Resolution| T |  Param.     | FLOPs   |  Energy Consumption |Top-1 Acc|
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:              |:---: |
| Spikingformer-8-384 | 224x224   | 4 |  16.81M     | 6.82G   | 4.69 mJ   |72.45  |
| Spikingformer-8-512 | 224x224   | 4 |  29.68M     | 11.09G  | 7.46 mJ   |74.79  |
| Spikingformer-8-768 | 224x224   | 4  |  66.34M     | 22.09G  | 13.68 mJ  |75.85  |


## Requirements
timm==0.5.4;
cupy==10.3.1;
pytorch==1.10.0+cu111;
spikingjelly==0.0.0.0.12;
pyyaml;

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train
### Training  on ImageNet
Setting hyper-parameters in imagenet.yml

```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Training  on CIFAR10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```

### Training  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
cd cifar10
python train.py
```

### Training  on DVS128 Gesture
```
cd dvs128-gesture
python train.py
```

### Training  on CIFAR10-DVS
```
cd cifar10-dvs
python train.py
```


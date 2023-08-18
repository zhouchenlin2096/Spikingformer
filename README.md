# Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network, [Arxiv 2023](https://arxiv.org/abs/2304.11954)
Spikingformer is a pure event-driven transformer-based spiking neural network (**75.85% top-1** accuracy on ImageNet-1K, **+ 1.04%** and **significantly reduces energy consumption by 57.34%** compared with Spikformer). To our best knowledge, this is the first time that **a pure event-driven transformer-based SNN** has been developed in in 2023/04.

**Note！！！** Welcome to try **the latest improved version** of Spikingformer : [Spikingformer-CML](https://github.com/zhouchenlin2096/Spikingformer-CML), which achieves the state-of-the-art performance on several datasets (eg. **77.64 % top-1** on ImageNet-1K with 224*224 inference resolution) in directly trained SNNs .


<p align="center">
<img src="https://github.com/zhouchenlin2096/Spikingformer/blob/master/imgs/Spikingformer-Architecture.png">
</p>

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

## Main results on ImageNet-1K

| Model               | Resolution| T |  Param.     | FLOPs   |  Power |Top-1 Acc| Download |
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:    |:---: |:---: |
| Spikingformer-8-384 | 224x224   | 4 |  16.81M     | 3.88G   | 4.69 mJ   |72.45  |       |
| Spikingformer-8-512 | 224x224   | 4 |  29.68M     | 6.52G  | 7.46 mJ   |74.79  |       |
| Spikingformer-8-768 | 224x224   | 4  |  66.34M     | 12.54G  | 13.68 mJ  |75.85  |   [here](https://pan.baidu.com/s/1LsECpFOxh30O3vHWow8OGQ) |

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

| Model               | T      |  Param.     |  CIFAR10 DVS Top-1 Acc  | DVS 128 Top-1 Acc|
| :---:               | :---:  | :---:       | :---:                   |:---:            |
| Spikingformer-2-256 | 10     |  2.57M      | 79.9                    | 96.2            |
| Spikingformer-2-256 | 16     |  2.57M      | 81.3                    | 98.3            |


## Requirements
timm==0.6.12; cupy==11.3.0; torch==1.14.0+cu116; spikingjelly==0.0.0.0.12; pyyaml; 

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

## Acknowledgement & Contact Information
Related project: [spikformer](https://github.com/ZK-Zhou/spikformer), [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [spikingjelly](https://github.com/fangwei123456/spikingjelly).

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact zhouchl@pcl.ac.cn or zhouchenlin19@mails.ucas.ac.cn.

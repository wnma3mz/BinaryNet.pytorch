# BNN.pytorch
Binarized Neural Network (BNN) for pytorch
This is the pytorch version for the BNN code, fro VGG and resnet models
Link to the paper: https://papers.nips.cc/paper/6573-binarized-neural-networks

The code is based on https://github.com/eladhoffer/convNet.pytorch
Please install torch and torchvision by following the instructions at: http://pytorch.org/
To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10

`data.py`: 加载数据集

`utils.py`: 可视化

`preprocess.py`: 对数据集进行预处理以及可视化预处理

`main_binary.py`: 标准的BNN

`main_binary_hinge.py`: 在标准BNN的基础上替换损失函数为hinge

`main_mnist.py`: 自定义的网络模型

`models/`: 定义网络模型

    `binarized_modules.py`: 实现BNN的模块，替换Linear和Conv层

    - 三种网络结构的binary实现

        `alexnet_binary.py`

        `resnet_binary.py`

        `vgg_cifar10_binary.py`
    
    - 网络结构的标准版本

        `resnet.py`

        `vgg_cifar10.py`

        `alexnet.py`

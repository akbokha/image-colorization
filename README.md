# image-colorization

This framework facilitates the training and evaluation of various deep neural networks for the task of image colorization.
In particular, it offers the following colorization models, features and evaluation methods:

<b>Colorization models</b>
- ResNet Colorization Network
- Conditional GAN (CGAN)
- U-Net

<b>Evaluation methods and metrics</b>
- The Mean Squared Error (MSE)
- The Mean LPIPS Perceptual Similarity (PS)
- Semantic Interpretability (SI)

## Prerequisites
The framework is implemented in Python (3.6) using PyTorch v1.0.1.  

Please consult `./env/mlp_env.yml` for a full list of the dependencies of the Conda environment that was used in the development of this framework. 
If Conda is used as as package and environment manager, one can use 
 `conda create --name myenv --file ./env/mlp_env.txt` to recreate the aforementioned environment. 
 
## Structure

- `train.py` - main entry point of the framework
- `src/options.py` - parses arguments (e.g. task specification, model options) 
- `src/main.py` -  set-up of task environment (e.g. models, dataset, evaluation method)
- `src/dataloaders.py` - downloads and (sub)samples datasets, and provides iterators over the dataset elements.
- `src/models.py` - contains the implementations of the model architectures
- `src/utils.py` - contains various helper functions and classes
- `src/colorizer.py` - trains and validates colorization models
- `src/classifier.py` - trains and validates image-classification models (used for SI)
- `src/eval_gen` - contains helper functions for the evaluation of model colorizations
- `src/eval_mse.py` - evaluates colorizations by MSE
- `src/eval_ps.py` - evaluates colorizations by the Mean LPIPS Perceptual Similarity (PS)
- `src/eval_si.py` - evaluates colorizations by Semantic Interpretability (SI)

## Usage

<b>Training of models</b> \
`python train.py [--option ...]` where the options are:

 option | description | type | oneOf | default 
--------|-------------|-------|-------|---------
 `seed` | random seed | `int` | not applicable | `0` 
 `task` | the task that should be executed | `str` |  `['colorizer', 'classifier', 'eval-gen', 'eval-si', 'eval-ps', 'eval-mse']` | `'colorizer'` 
 `experiment-name` | the name of the experiment | `str` |  not applicable| `'experiment_name'`  
 `model-name` | colorization model architecture that should be used | `str` | `['resnet', 'unet32', 'unet224', 'nazerigan32', 'nazerigan224' 'cgan']` | `'resnet'`
`model-suffix` | colorization model name suffix | `str`  | not applicable  | not applicable 
`model-path` | path for the pretrained models | `str` | not applicable | `'./models'`
`dataset-name` | the dataset to use | `str` | `['placeholder', 'cifar10', 'places100', 'places205', 'places365']` | `'placeholder'`
`dataset-root-path` | dataset root path | `str` | not applicable | `'./data'`
`use-dataset-archive` | load dataset from TAR archive | `str2bool` | `[True, False]` | `False`
`output-root-path` | path for output (e.g. model  weights, stats, colorizations) | `str` | not applicable | `'./output'`
`max-epochs` | maximum number of epochs to train for | `int` | not applicable | `5`
`train-batch-size` | training batch size | `int` | not applicable | `100`
`val-batch-size` | validation batch size | `int` | not applicable | `100`
`batch-output-frequency` | frequency with which to output batch statistics | `int` | not applicable | `1`
`max-images` | maximum number of images from the validation set to be saved (per epoch) | `int` | not applicable | `10`
`eval-root-path` | the root path for evaluation images | `str` | not applicable | `'./eval'`
`eval-type` | the type of evaluation task to perform | `str` | `['original, 'grayscale', 'colorized']` | `'original'`

So one could for example train a cgan colorization model on the places365 dataset for 100 epochs by running:
```bash
python train.py \
  --experiment-name cgan_experiment001 \  
  --model-name cgan \        
  --dataset-name places365 \ 
  --max-epochs 100 \
  --train-batch-size 16 \
  --val-batch-size 16 \
```

## Colorization Task
The task of colorizing a image can be considered a pixel-wise regression problem where the model input _<b>X</b>_ is a _1xHxW_ tensor containing the pixels of the grayscale imageand the model output _<b>Y'</b>_ a tensor of shape _nxHxW_ that represents the predicted colorization information. 
Specifically, the task aims to discover a mapping _F: <b>X</b> &rarr; <b>Y</b>'_ that plausibly predicts the colorization given the greyscale input.

<img src="/media/ImgPipeline.png"/>

The [CIE _L\*a\*b\*_ colour space](https://en.wikipedia.org/wiki/CIELAB_color_space) lends itself well to this task since the _L_ channel depicts the brightness of the image (_<b>X</b>_ above) and the image colour is fully captured in the remaining _a_ and _b_ channels (_<b>Y'</b>_ above). The _L\*a\*b\*_ colour model also has the advantage of being inspired by human colour perception, meaning that distances in _L\*a\*b\*_
space can be expected to be correlated with changes in human colour perception. The final output colorized image is
created by recombining the input L layer with the predicted _a_ and _b_ layers.

## Colorization Models
Three colorization architectures are currently supported in the framework.

#### ResNet Colorization Network

This architecture consists of a CNN that starts out with a set of convolutional layers which aim to extract low-level and semantic features from the set of input images, inspired by how representations are learned in [Learning Representations for Automatic Colorization](https://arxiv.org/abs/1603.06668?utm_source=top.caibaojian.com/92010).
Based on the same idea as behind the VGG-16-Gray architecture in this paper, a modified version of the image classification network that is [ResNet-18](https://arxiv.org/abs/1512.03385) is used as a means to learn representations from a set of images. In particular, the network is modified in such a way that it accepts greyscale images and in addition, the network is truncated to six layers.
This set of layers is used to extract features from the images that are represented by their lightness channels. Subsequently a series of deconvolutional layers is applied to increase the spacial resolution of (i.e. 'upscale') the features. This up-scaling of features learned in a network is inspired by the 'upsampling' of features in the colorization network of [Let There Be Color!](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)

#### U-Net
This network is inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) where direct connections are added between contracting and expanding layers of equal size to prevent the loss of spatial context of the original image throughout the layers.
In [Image Colorization with Generative Adversarial Networks](https://arxiv.org/abs/1803.05400) an approach is proposed that uses such a network for colorization since the preservation of the original greyscale image is of particular importance to this task. 
The network implemented in this paper has the same architecture as the one presented in the original U-Net paper, modified to take 224x224 inputs. 
Non-linearities are introduced by following convolutional and deconvolutional layers with leaky ReLUs with slope of 0.2. Furthermore batch normalisation is applied after every layer.

<b>Conditional GAN (CGAN)</b>

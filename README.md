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
`eval-root-path` | the root path for evaluation images | `str` | not applicable | `'./eval`
`eval-type` | the type of evaluation task to perform | `str` | `['original, 'grayscale', 'colorized']` | `'original'`

So one could for example train a cgan colorization on the places365 dataset for 100 epochs by running:
```bash
python train.py \
  --experiment-name cgan_experiment001 \  
  --model-name cgan \        
  --dataset-name places365 \ 
  --max-epochs 100 \
  --train-batch-size 16 \
  --val-batch-size 16 \
```

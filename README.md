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

### Prerequisites
The framework is implemented in Python (3.6) using PyTorch v1.0.1.  

Please consult `./env/mlp_env.yml` for a full list of the dependencies of the Conda environment that was used in the development of this framework. 
If Conda is used as as package and environment manager, one can use 
 `conda create --name myenv --file ./env/mlp_env.txt` to recreate the aforementioned environment. 
 
### Structure

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
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

Please consult `./env/mlp_env.yml` for a full list of the dependencies of the Conda environment that was used in the development of this framework. \
If Conda is used as as package and environment manager, one can use 
 `conda create --name myenv --file ./env/mlp_env.txt` to recreate the aforementioned environment. 
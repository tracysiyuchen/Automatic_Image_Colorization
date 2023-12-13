# Image Colorization Project

This project focuses on colorizing grayscale images using deep learning models. It allows users to choose between models for image colorization.

## Prerequisites

Before you start, ensure you have met the following requirements:

- You have installed [Python 3.6](https://www.python.org/) or later.
- You have [Conda](https://www.anaconda.com/products/individual) installed on your machine.

## Installation

To set up your environment and run the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/tracysiyuchen/Automatic_Image_Colorization.git
   cd Automatic_Image_Colorization
   ```

2. Create and activate the Conda environment:
   ```sh
   conda env create -f environment.yml
   conda activate color
   ```

3. Download the dataset from the Kaggle dataset [Image Colorization](https://www.kaggle.com/datasets/shravankumar9892/image-colorization/data)  and place it in the `data` folder. Make sure you have the following structure:
   ```plaintext
   data/
   ├── gray_scale.npy
   └── ab/
       └── ab1.npy
   ```

4. You are now ready to run the project. To start training the model, use:
   ```sh
   python train.py
   ```


## Usage
The main script for training the model is `train.py`. Below are the flags and options you can use:

- `--gray-path`: Path to the gray scale images. Default is `data/gray_scale.npy`.
- `--ab-path`: Path to the ab channel images. Default is `data/ab/ab1.npy`.
- `--num-images`: Number of images to load. Default is `1000`.
- `--split-ratios`: Train, validation, and test split ratios. Default is `[0.7, 0.1, 0.2]`.
- `--batch-size`: Batch size for training. Default is `64`.
- `--epoch`: Number of epochs for training. Default is `10`. 
- `--lr`: Learning rate for training. Default is `0.001`
- `--device`: device used for training(cpu, cuba:0, etc). Default is  `cpu`  
- `--seed`: Random seed for reproducibility. Default is  `None`  
- `--model`: The model to use for training. Options are `cnn`, `mobilenet` or `gan`. Default is `cnn`.

You can specify the model type using the `--model` flag. For example:
- For CNN: 
  ```sh
  python train.py --model cnn
  ```
- For MobileNet: 
  ```sh
  python train.py --model mobilenet
  ```
- For cGAN:
  ```sh
  python train.py --model gan
  ```

The training progress and results will be displayed in the console.

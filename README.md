# DeblurGAN - Image Deblurring
Unofficial implementation of the DeblurGAN model. With this repository you will be able to train the model with your own dataset from scratch.
Unofficial implementation of the DeblurGAN model based on the paper [DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks](https://arxiv.org/pdf/1609.04802v5). With this repository, you will be able to implement the model from scratch using your own dataset. The steps to train and test the model, along with the project details, are explained below.

## Dataset preparation

To prepare the dataset, you should first run the file `dataset.py`, setting the parameters at the beginning of the file as desired. These parameters are:

- **dataset_path**: specify the paths where the images to be used for training are located, separated by the character `;`.
- **train_split**: defines the ratio for splitting images into training and validation sets. For example, if there are 100 images and this parameter is set to 0.8, then 80% of the images will be used for training and the remaining 20% for validation.
- **min_width** and **min_height**: these two parameters set the minimum width and height of the images to be taken into account. All images with smaller dimensions will be discarded.

The script lists all image paths in the given folders that meet the size requirements, shuffles them randomly, and then splits them into training and validation sets according to the **train_split** parameter. Finally, two _JSON_ files will be generated: one containing the training image paths and the other containing the validation image paths. These _JSON_ files will be used during training to load the images.

## Train the model
To train the model, you can adjust the parameters and hyperparameters as needed. In my experiments, I used the same values as in the original paper, but I implemented a different approach to blur the images. Specifically, I use the [OpenCV](https://opencv.org/) function [GaussianBlur](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gae8bdcd9154ed5ca3cbc1766d960f45c1), selecting a random kernel size (3, 5, or 7) and a random sigma value (between 2.0 and 9.0) each time an image is loaded. This introduces variability, allowing the model to learn to generalize better, since the same image can appear with different levels of blur at each iteration. To start the training, run:
```
python train.py
```
When the training starts, a folder is created. The folder name consists of the name provided in the parameters, followed by relevant training parameter information, such as the batch size, learning rate and the number of residual blocks added to the network. This way, you can always identify the parameter values used for the training. In my case, the name of my training folder is ```deblurGAN_bs1_lr0.0001_numresblocks9_lambdaG100_lambdaD10/```. Five txt files are stored in this folder, containing the values of the generator loss and the discriminator accuracy at each epoch. Additionally, within this folder, a subfolder called ```checkpoints/``` is created to store the model every 10 epochs. I am providing the generator model I trained, along with my txt files containing the loss and accuracy values.

## Test the model
To test the model, specify the checkpoint to be evaluated in the parameters, along with the name of the folder containing the model file. Place the images you want to deblur in the `test_images/original/` folder and run:
```
python test_images.py
```
The resulting deblurred images will be saved in the `test_images/deblurred/` folder. As an example, you can check the folder included in this repository, which contains an image I deblurred.

The same process applies to videos by running:
```
python test_videos.py
```

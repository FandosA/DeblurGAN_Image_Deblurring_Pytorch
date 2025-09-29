# DeblurGAN - Image Deblurring
Unofficial implementation of the DeblurGAN model. With this repository you will be able to train the model with your own dataset from scratch.
Unofficial implementation of the DeblurGAN model based on the paper [DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks](https://arxiv.org/pdf/1609.04802v5). With this repository, you will be able to implement the model from scratch using your own dataset. The steps to train and test the model, along with the project details, are explained below.

## Dataset preparation

To prepare the dataset, you should first run the file `dataset.py`, setting the parameters at the beginning of the file as desired. These parameters are:

- **dataset_path**: specify the datasets you want to use for training, separated by the character `;`.
- **train_split**: defines the ratio for splitting images into training and validation sets. For example, if there are 100 images and this parameter is set to 0.8, then 80% of the images will be used for training and the remaining 20% for validation.
- **min_width** and **min_height**: these two parameters set the minimum width and height of the images to be taken into account. All images with smaller dimensions will be discarded.

The script lists all image paths in the given folders that meet the size requirements, shuffles them randomly, and then splits them into training and validation sets according to the **train_split** parameter. Finally, two JSON files will be generated: one containing the training image paths and the other containing the validation image paths. These _JSON_ files will be used during training to load the images.

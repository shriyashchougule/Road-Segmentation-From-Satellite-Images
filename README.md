# Road-Segmentation-From-Satellite-Images

## Network Architecture:

In order to train a network for satellite road segmentation task, I preferred to use a known architecture for developing the first prototype.

Also wanted to use an architecture for which pre-trained weights are available to help in and converging the network faster. I decided to use Keras framework which has several popular deep learning networks with  pre trained weights on ImageNet classification challenge. I decided to use Dense201 network as a base network, which apparently evaluates better in terms of parameters vs Top1 Accuracy tradeoff.

For general segmentation tasks, UNet style connection patterns are a proven choice to go ahead for the first prototype. Using a dense net as a base network (removing the classification head), I build a UNet style connection on top of the base DenseNet. I will refer to this new structure as DenseUnet. The UNet style connection I have used was borrowed from another github repo (forgot to save the link to it).

All networks in keras are trained for 224x224 images, So decided to have 224x224 subimages cropped out of the original image, and make predictions on it. I use Dice loss which is proven to be handle class imbalance well for segmentation task. I have borrowed the dice loss implementation from [here](https://github.com/Paulymorphous/skeyenet/blob/master/Src/loss_functions.py).
## Data Cleaning:

I applied two levels of cleaning.\
A. Ignored images which had white patches covering over 20% of image area (or pixels).\
B. Ignored images for which ground truth pixels are less than 2 percent of total area (pr pixels)  

Also I observed that some missing or wrong labels are also present. And some structures which resemble roads but are not. All these will confuse the network, which is not conducive to network convergence. 
Data Augmentation:

I split the training data (804 images ) into training set (744 images) and validation set (60) images before applying data augmentation.

For data augmentation I used crops from training data at fixed intervals and also applied vertical and horizontal flipping. The data augmentation from keras could have been handy, But I avoided with suspicion that it would distort the ground truth pixels, as I deemed the keras fill methods not suitable (I have not verified it)

I created an HDF5 file from both the training (8GB) and validation data (700MB), as it would be convenient and faster to upload the hdf5 files than individual images. This will also speed up training time (memory reads are faster with hdf5 file). For taking advantage of HDF5 files, I have written custom Data generators. Training and Validation HDF5 file are [here] (https://drive.google.com/drive/folders/1TU6NG-83GYDfknMGkJj61EMJNbFQvynp?usp=sharing)

## GitRepository Files 
**custom_batch_generator_hpf5.py** : Has code for my custom Data generator\
**generate_dataset_using_subimages_v2.py**: Script used to create training and validation datasets hdf5 files respectively.  Data cleaning and Data augmentation are also done in this script.\
**Verify_generator.py** is a support script to test data generators.\
**Train DenseUnet and Save weights.ipynb** Is python notebook used to train the networks\
**Predictions on Test Set.ipynb** has prediction and performance metrics evaluated.

## Evaluation on Test Set:
Average Values
IoU: 0.5824341424308911, Accuracy: 0.9628995555555555, Recall: 0.7538033794836566, Precision: 0.7186851959629647\
*Model* [roadsegDenseUnet.h5](https://drive.google.com/file/d/1_jCy2RUCS9PyEYe2Aat5G6NPvcjeyw4-/view?usp=sharing) is used for performance evaluation on test set.

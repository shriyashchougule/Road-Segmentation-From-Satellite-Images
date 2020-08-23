import cv2
import numpy as np
import os
import random
import h5py
import sys

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return image

def subimages_croped_from_given_images(datasetPaths, subimage_width=224, subimage_height=224, stride = 224):
    image_list = list()
    mask_list = list()
    ignored = 0
    potentialImages = 0
    emptyImages = 0

    #hf_images = h5py.File('TrainImages.h5', 'w')
    #hf_masks = h5py.File('TrainMasks.h5', 'w')
    hf_images = h5py.File('ValidationImages.h5', 'w')
    hf_masks = h5py.File('ValidationMasks.h5', 'w')
    #hf_images.create_dataset('my_images', (178689, 256, 256, 3), dtype='uint8')
    #hf_masks.create_dataset('my_masks', (178689, 256, 256), dtype='uint8')

    index = 0
    for imagePath, maskPath in datasetPaths:
        print(imagePath, maskPath)
        image = cv2.imread(imagePath)
        mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
        if ((image is not None) and (mask is not None)):
            width, height, channels = image.shape
            for row in range(0, height - subimage_height, stride):
                for col in range(0, width - subimage_width, stride):
                    subimage = np.array(image[ row:row+subimage_height, col:col+subimage_width ])
                    submask = np.array(mask[ row:row+subimage_height, col:col+subimage_width ])

                    if np.any(submask):
                        background, road = np.unique(submask, return_counts=True)[1]
                        # ignore where road pixle fall below 1% of total pixels
                        if road/background < 0.01:
                            ignored += 1
                            continue
                        # ignore if image has more than 20% of white patch
                        if np.sum(np.all(subimage==255, axis=2))/(subimage_width*subimage_height) > 0.2:
                            ignored += 1
                            continue
                        potentialImages += 3

                        hf_images.create_dataset(name = str(index), data = subimage, shape = (subimage_width,subimage_height,3), compression="gzip", compression_opts=9)
                        hf_masks.create_dataset(name = str(index), data = submask, shape = (subimage_width,subimage_height), compression="gzip", compression_opts=9)
                        index += 1
                        # vertical flip
                        hf_images.create_dataset(name = str(index), data = cv2.flip(subimage, 0), shape = (subimage_width,subimage_height,3), compression="gzip", compression_opts=9)
                        hf_masks.create_dataset(name = str(index), data = cv2.flip(submask, 0), shape = (subimage_width,subimage_height), compression="gzip", compression_opts=9)
                        index += 1
                        # horizontal flip
                        hf_images.create_dataset(name = str(index), data = cv2.flip(subimage, 1), shape = (subimage_width,subimage_height,3), compression="gzip", compression_opts=9)
                        hf_masks.create_dataset(name = str(index), data = cv2.flip(submask, 1), shape = (subimage_width,subimage_height), compression="gzip", compression_opts=9)
                        index += 1


                    else:
                        emptyImages+= 1
        print("\npotential images: {}, index: {}, empty images: {},  ignored images: {}".format(potentialImages, index, emptyImages, ignored))
        hf_images.flush()
        hf_masks.flush()
    hf_images.close()
    hf_masks.close()
    return (image_list, mask_list)

def get_correspoding_image_and_mask_paths(imageDir, maskDir):
    pathMapping = list()
    imageNameList = os.listdir(imageDir)
    for maskName in os.listdir(maskDir):
        if maskName in imageNameList:
            imagePath = os.path.join(imageDir, maskName)
            maskPath = os.path.join(maskDir, maskName)
            pathMapping.append((imagePath, maskPath))
    return pathMapping

if __name__ == "__main__":
    #imageDir = "/home/calm/Pictures/images"
    #maskDir =  "/home/calm/Pictures/labels"
    trainimageDir = "/home/calm/Downloads/road_segmentation_ideal/training/input"
    trainmaskDir =  "/home/calm/Downloads/road_segmentation_ideal/training/output"

    valimageDir = "/home/calm/Downloads/road_segmentation_ideal/validation/input"
    valmaskDir =  "/home/calm/Downloads/road_segmentation_ideal/validation/output"

    #pathMappings = get_correspoding_image_and_mask_paths(trainimageDir, trainmaskDir)
    pathMappings = get_correspoding_image_and_mask_paths(valimageDir, valmaskDir)
    subimages, submasks = subimages_croped_from_given_images(pathMappings)
    print("subimages {} submask {}".format(len(subimages), len(submasks)))

    #hf_images = h5py.File('images.h5', 'w')
    #hf_masks = h5py.File('masks.h5', 'w')
    #hf_images.create_dataset('my_images', data = np.array(subimages))
    #hf_masks.create_dataset('my_masks', data = np.array(submasks))
    #for i,m in zip(subimages, submasks):
    #    cv2.imshow("img", i)
    #    cv2.imshow("mask",m)
    #    cv2.waitKey(5)
    print("done")
    #hf_images.close()
    #hf_masks.close()

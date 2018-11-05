import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pickle

def merge_annotation(image_array, reshaped_annotation, size):
    assert reshaped_annotation.ndim == 2
    assert size % 2 == 0
    assert image_array.dtype == np.uint8
    height, width = image_array.shape[0], image_array.shape[1]
    annotation = np.zeros(reshaped_annotation.shape, dtype=np.int32)
    annotation[:,0] = np.round(reshaped_annotation[:,0]*width).astype(np.int32)
    annotation[:,1] = np.round(reshaped_annotation[:,1]*height).astype(np.int32)
    
    with Image.fromarray(image_array) as img:
        draw = ImageDraw.Draw(img)
        for i in range(annotation.shape[0]):
            draw.line((annotation[i,0]-size//2, annotation[i,1], annotation[i,0]+size//2, annotation[i,1]),
                        fill=(0,192,128), width=3)
            draw.line((annotation[i,0], annotation[i,1]-size//2, annotation[i,0], annotation[i,1]+size//2),
                        fill=(0,192,128), width=3)
        merged = np.asarray(img) / 255.0
    return merged

from train import create_resnet, CatGenerator

def plot_grandtruth(is_train):
    gen = CatGenerator()

    X, y_true = next(gen.flow_from_directory(16, is_train, False, use_data_augmentation=False))

    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.02, wspace=0.02, top=0.95, bottom=0.05, left=0.05, right=0.95)
    for i in range(16):
        ax = plt.subplot(4, 4, i+1)
        img = (X[i] * 255.0).astype(np.uint8)
        annotation = y_true[i].reshape(9,2 )
        merged = merge_annotation(img, annotation, 24)
        ax.imshow(merged)
        ax.axis("off")
    plt.show()

def plot(is_train):
    model = create_resnet()
    model.load_weights(f"cats_weights.hdf5")

    gen = CatGenerator()

    X, y_true = next(gen.flow_from_directory(18, is_train, False, use_data_augmentation=False))
    X = X[1::2]
    y_pred = model.predict(X)

    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.02, wspace=0.02, top=0.95, bottom=0.05, left=0.05, right=0.95)
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        img = (X[i] * 255.0).astype(np.uint8)
        annotation = y_pred[i].reshape(9,2 )
        merged = merge_annotation(img, annotation, 24)
        ax.imshow(merged)
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    #plot_grandtruth(False)
    plot(True)

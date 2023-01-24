import numpy as np
from matplotlib import pyplot as plt
from utils import default_loader
from config.settings import NetworkSettings


if __name__ == '__main__':

    train_images = default_loader(NetworkSettings.Training.DATA)
    train_masks = default_loader(NetworkSettings.Training.MASKS)

    for image, mask in zip(train_images, train_masks):
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.imshow(image, cmap='jet')
        ax1.set_title('PMSE Images')
        ax1.axis('off')

        ax2.imshow(mask)
        ax2.set_title('Mask')
        ax2.axis('off')

    plt.show()
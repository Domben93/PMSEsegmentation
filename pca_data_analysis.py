import os.path
import random
import numpy as np
import torch
import torch.nn as nn
import glob
import math
import models.base_model
import utils.transforms
from utils.dataset import PMSE_Dataset
from utils import transforms as t
from torch.utils.data import DataLoader
from utils.utils import *
from config.settings import NetworkSettings as Settings
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import k_means, DBSCAN
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from numpy import ndarray
from models.unets import UNet


def save_images(path: str, data: PMSE_Dataset, data_info, v_min, v_max):
    files = glob.glob(path + '/*')
    for f in files:
        os.remove(f)

    dataloader = DataLoader(data, batch_size=1)
    numerated_images = generate_names(data_info)
    for num, (img, mask, _) in enumerate(dataloader):
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0.1, 'hspace': 0.0})
        ax1.imshow(img.squeeze(), cmap='jet', vmin=v_min, vmax=v_max)
        ax1.axis('off')
        ax2.imshow(mask.squeeze(), cmap='jet')
        ax2.axis('off')
        plt.title(numerated_images[num])
        plt.savefig(os.path.join(path, numerated_images[num]))
        plt.close(fig)


def generate_names(data_info):
    enumerated_img_list = []
    img_num = 1
    sub_img = 1

    for num, info in enumerate(data_info['image_name']):
        if num == 0:
            enumerated_img_list.append(str(img_num) + '-' + str(sub_img))
            last_info = info
        else:
            if info != last_info:
                sub_img = 1
                img_num += 1
            else:
                sub_img += 1

            enumerated_img_list.append(str(img_num) + '-' + str(sub_img))
        last_info = info

    return enumerated_img_list


class PmseDataAnalysis:

    def __init__(self,
                 data_path: str,
                 label_path: str):

        pair_compose = t.PairCompose([
            [t.QuasiResize(size=[64, 64], max_scaling=2, padding_mode='constant'),
             t.QuasiResize(size=[64, 64], max_scaling=2, padding_mode='constant')],
            [t.ToGrayscale(output_channels=3), None],
            [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)]
        ])

        self._data = PMSE_Dataset(data_path, label_path,
                                  transform=pair_compose,
                                  square_split=True,
                                  percent_overlap=.0,
                                  last_min_overlap=0,
                                  lock_vertical=True)

        self._dataloader = DataLoader(self._data, batch_size=len(self._data))

        self.img_data, self.mask_data, self.data_info = next(iter(self._dataloader))
        self.im_squeezed = self.img_data[:, 0, :, :].view(len(self._data), 4096)

        self.model_vector = None

    def PCA_datareduction(self, X: ndarray, n_components: int = 2, solver: str = 'auto'):
        pca = PCA(n_components=n_components, svd_solver=solver)

        reduced_data = pca.fit_transform(X)

        return reduced_data

    def TSNE(self, X: ndarray,
             n_components: int = 2,
             l_rate: int = 'auto',
             perplexity: int = 10,
             n_iter: int = 1000) -> ndarray:
        t_sne = TSNE(n_components=n_components,
                     learning_rate=l_rate,
                     init='random',
                     perplexity=perplexity,
                     n_iter=n_iter).fit_transform(X)
        return t_sne

    def last_layer_vectors(self, model: models.base_model.BaseModel, weights) -> ndarray:
        model.load_state_dict(weights)
        device = torch.device("cuda")
        model.to(device)

        vectors = []

        self._dataloader = DataLoader(self._data, batch_size=1)

        for data in self._dataloader:
            im, mask, info = data
            im = im.to(device)

            with torch.no_grad():
                res = model(im)

                vectors.append(res.cpu().detach().view(res.shape[-2] * res.shape[-1]).numpy())

                # print(vectors[-1])

        return np.array(vectors)

    def save_images(self, path):
        save_images(path=path, data=self._data, data_info=self.data_info, v_max=self._vmax, v_min=self._vmin)


class EmptyActivation(nn.Module):
    def __init__(self):
        super(EmptyActivation, self).__init__()

    def forward(self, input):
        return input


if __name__ == '__main__':

    config_path = 'models\\options\\unet_config.ymal'

    config = load_yaml_as_dotmap(config_path)

    train = PmseDataAnalysis(os.path.join(config.dataset.path_train, 'data'),
                               os.path.join(config.dataset.path_train, 'label'))

    val = PmseDataAnalysis(os.path.join(config.dataset.path_validate, 'data'),
                               os.path.join(config.dataset.path_validate, 'label'))

    test = PmseDataAnalysis(os.path.join(config.dataset.path_test, 'data'),
                               os.path.join(config.dataset.path_test, 'label'))

    unet = UNet(3, 1, 32, activation_func=EmptyActivation)

    vector_data_train = train.last_layer_vectors(unet,
                                                   torch.load(
                                                       'weights/lr_0.008_wd_0.005_betas_0.9-0.999_momentum_0.9_freezed-None_0.pt')
                                                   ['model_state'])

    vector_data_val = val.last_layer_vectors(unet,
                                           torch.load(
                                               'weights/lr_0.008_wd_0.005_betas_0.9-0.999_momentum_0.9_freezed-None_0.pt')
                                           ['model_state'])
    vector_data_test = test.last_layer_vectors(unet,
                                           torch.load(
                                               'weights/lr_0.008_wd_0.005_betas_0.9-0.999_momentum_0.9_freezed-None_0.pt')
                                           ['model_state'])

    pca_train = train.PCA_datareduction(vector_data_test, 50)#train.TSNE(train.PCA_datareduction(train.im_squeezed, 50), 2, perplexity=5)
    pca_val = val.PCA_datareduction(vector_data_val, 50)
    pca_test = test.PCA_datareduction(vector_data_test, 50)

    numbers_train = generate_names(train.data_info)
    numbers_val = generate_names(val.data_info)
    numbers_test = generate_names(test.data_info)

    fig, ax = plt.subplots(figsize=(10, 10))

    for n, (i, j) in enumerate(zip(pca_train[:, 0], pca_train[:, 1])):

        ax.scatter(i, j, color='red')
        ax.annotate(numbers_train[n], xy=(i, j), color='black',
                    fontsize="large", weight='heavy',
                    horizontalalignment='center')

    for n, (i, j) in enumerate(zip(pca_val[:, 0], pca_val[:, 1])):

        ax.scatter(i, j, color='blue')
        ax.annotate(numbers_val[n], xy=(i, j), color='black',
                    fontsize="large", weight='heavy',
                    horizontalalignment='center')

    for n, (i, j) in enumerate(zip(pca_test[:, 0], pca_test[:, 1])):
        ax.scatter(i, j, color='green')
        ax.annotate(numbers_test[n], xy=(i, j), color='black',
                    fontsize="large", weight='heavy',
                    horizontalalignment='center')

    plt.show()
    """
    data_loader = DataLoader(pmse_train_data, batch_size=1)


    mean, std = dataset_mean_and_std(data_loader)

    min_ = - std * 3
    max_ = std * 3
    im_size = 64

    transforms = t.PairCompose([[t.Standardize(mean=mean, std=std), None],
                                [t.Normalize((0, 255), (min_, max_)), None],
                                [t.ToTensor(), t.ToTensor(zero_one_range=False)]])

    pmse_train_data = PMSE_Dataset(Settings.CompleteSet.DATA,
                                   Settings.CompleteSet.MASKS,
                                   transform=transforms,
                                   disable_warnings=True)

    im1, msk1, _ = pmse_train_data[16]
    im2, msk2, _ = pmse_train_data[26]

    print(im1.numpy().squeeze().shape)
    fig1, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(im1.numpy().squeeze(), cmap='jet', vmax=1, vmin=0)
    ax2.imshow(im2.numpy().squeeze(), cmap='jet', vmax=1, vmin=0)

    fig2, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(msk1.numpy().squeeze(), cmap='jet')
    ax2.imshow(msk2.numpy().squeeze(), cmap='jet')


    plt.show()
    """

    """
    data_analysis = PmseDataAnalysis(data_path=Settings.CompleteSet.DATA,
                                     label_path=Settings.CompleteSet.MASKS,
                                     data_split_overlap=.0)

    #tsne = data_analysis.TSNE(data_analysis.PCA_datareduction(data_analysis.im_squeezed, 50), 2, perplexity=5)

    img_numbers = generate_names(data_analysis.data_info)

    #plt.scatter(tsne[:, 0], tsne[:, 1])
    #plt.show()

    unet = UNet(3, 1, 32, activation_func=EmptyActivation)

    vector_data = data_analysis.last_layer_vectors(unet,
                                                   torch.load(
                                                       'weights\\UNet_64x64_lr0'
                                                       '.001_freezed01235678_alldata_for_analysis.pt')
                                                   ['model_state'])

    pca = data_analysis.PCA_datareduction(vector_data, 2)
    #tsne = data_analysis.TSNE(data_analysis.PCA_datareduction(vector_data, 50), 2, perplexity=5, n_iter=2000)
    #plt.scatter(tsne[:, 0], tsne[:, 1])
    #plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))

    for n, (i, j) in enumerate(zip(pca[:, 0], pca[:, 1])):

        ax.plot(i, j, markersize=30)
        ax.annotate(img_numbers[n], xy=(i, j), color='black',
                    fontsize="large", weight='heavy',
                    horizontalalignment='center')

    plt.legend()
    plt.show()
    """
    """
    pmse_train_data = PMSE_Dataset(Settings.CompleteSet.DATA,
                                   Settings.CompleteSet.MASKS,
                                   transform=t.ToTensor(),
                                   disable_warnings=True)

    data_loader = DataLoader(pmse_train_data, batch_size=1)

    mean, std = dataset_mean_and_std(data_loader)

    min_ = - std * 3
    max_ = std * 3

    v_min, v_max = 0, 1

    split_overlap = .3

    pair_compose = t.PairCompose([
        [t.Standardize(mean=mean, std=std), t.MaskClassReduction([0, 1, 2, 3], [0, 1], 0)],
        [t.Normalize((0, 255), (min_, max_)), None],
        [t.ToTensor(), t.ToTensor(zero_one_range=False)],
        [t.QuasiResize(size=[64, 64], max_scaling=2, padding_mode='replicate'), t.QuasiResize(size=[64, 64], max_scaling=2, padding_mode='replicate')],
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)]
    ])

    pmse_train_data = PMSE_Dataset(Settings.CompleteSet.DATA,
                                   Settings.CompleteSet.MASKS,
                                   transform=pair_compose,
                                   square_split=True,
                                   percent_overlap=split_overlap)

    print(f'Number of images: {len(pmse_train_data)}')

    data_loader = DataLoader(pmse_train_data, batch_size=len(pmse_train_data))

    im, mask, data_info = next(iter(data_loader))

    im_squeezed = im.view(len(pmse_train_data), 4096)

    save_transform = t.PairCompose([[t.Standardize(mean=mean, std=std), t.Transpose((2, 0, 1))],
                                    [t.Normalize((0, 255), (min_, max_)), None],
                                    [t.ToTensor(), t.ToTensor(zero_one_range=False)],
                                    [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)]
                                    ])

    save_image_data = PMSE_Dataset(Settings.CompleteSet.DATA,
                                   Settings.CompleteSet.MASKS,
                                   transform=save_transform,
                                   square_split=True,
                                   percent_overlap=split_overlap)

    #save_images("C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\splitimages", save_image_data, data_info)

    pca = PCA(n_components=2)
    red_data = pca.fit_transform(im_squeezed)

    centroids, labels, _ = k_means(red_data, n_clusters=5)

    x = red_data[:, 0]  # first PCA component
    y = red_data[:, 1]  # second PCA component

    fig, ax = plt.subplots(figsize=(10, 10))

    #ax.plot(x, y, 'bo', markersize=23)

    ## controls the extent of the plot.
    offset = 1.0
    ax.set_xlim(min(x) - offset, max(x) + offset)
    ax.set_ylim(min(y) - offset, max(y) + offset)
    colour_list = ['r', 'b', 'g', 'c', 'k', 'y']
    # loop through each x,y pair
    image_name = data_info['image_name']
    image_num = 1
    part_img_num = 1
    for n, (i, j) in enumerate(zip(x, y)):
        if n == 0:
            image_num = 1
            part_img_num = 1
        else:
            if image_name[n-1] != image_name[n]:
                image_num += 1
                part_img_num = 1
            else:
                part_img_num += 1
        #ax.plot(i, j, colour_list[labels[n]] + 'o', markersize=30)
        ax.annotate(str(image_num) + '-' + str(part_img_num), xy=(i, j), color='black',
                    fontsize="large", weight='heavy',
                    horizontalalignment='center')

    plt.legend()

    enumerated_img_list = generate_names(data_info)

    db = DBSCAN(eps=0.8, min_samples=10).fit(red_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for num, (k, col) in enumerate(zip(unique_labels, colors)):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        
        #xy = red_data[class_member_mask & core_samples_mask]
        #plt.plot(
        #    xy[:, 0],
        #    xy[:, 1],
        #    "o",
        #    markerfacecolor=tuple(col),
        #    markeredgecolor="k",
        #    markersize=14,
        #)
        
        xy = red_data[class_member_mask] # & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=10,
            label='Cluster' + str(num)
        )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.legend()
    index_list = []

    for label in (set(labels)):
        index_list.append(list(np.argwhere(labels == label)))

    fig, axes = plt.subplots(10, len(set(labels))*2, figsize=(10, 16), gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                             sharex=True)
    
    for m in range(len(set(labels))):

        cluster_image_index = random.sample(index_list[m], 10)

        for n, index in enumerate(cluster_image_index):
            plt.setp(axes[n, m * 2].get_xticklabels(), visible=False)
            plt.setp(axes[n, m * 2].get_yticklabels(), visible=False)
            plt.setp(axes[n, (m * 2) + 1].get_xticklabels(), visible=False)
            plt.setp(axes[n, (m * 2) + 1].get_yticklabels(), visible=False)
            axes[n, m * 2].tick_params(axis='both', which='both', length=0)
            axes[n, m * 2].imshow(im[index, 0, :, :].squeeze(), cmap='jet', vmin=v_min, vmax=v_max)
            axes[n, (m * 2) + 1].tick_params(axis='both', which='both', length=0)
            axes[n, (m * 2) + 1].imshow(mask[index, 0, :, :].squeeze(), cmap='jet')
            axes[n, m * 2].set_ylabel(enumerated_img_list[int(index)])

            if n == 0:
                if (m + 1) == len(set(labels)):
                    axes[n, m * 2].set_title('Cluster Outliers')
                    axes[n, (m * 2) + 1].set_title('Mask')
                else:
                    axes[n, m * 2].set_title('Cluster ' + str(m + 1) + ' PMSE')
                    axes[n, (m * 2) + 1].set_title('Mask')

    plt.show()

    """

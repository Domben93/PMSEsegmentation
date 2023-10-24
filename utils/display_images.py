import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torchvision.transforms.functional import resize
import torch
import matplotlib as mpl
import matplotlib
import matplotlib.gridspec as gridspec


def plot_predictions(path, sub_folders, name_list, image_list, save_name, figsize=(8, 8)):
    fig, axes = plt.subplots(len(image_list), len(sub_folders) + 2, figsize=figsize,
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.0})
    folder = glob.glob(path + '/' + sub_folders[0])
    org_image_path = glob.glob(folder[0] + '/' + 'images' + '/*')
    org_mask_path = glob.glob(folder[0] + '/' + 'labels' + '/*')

    n = 0
    for i, img in enumerate(org_image_path):
        if (i + 1) in image_list:
            plt.setp(axes[n, 0].get_xticklabels(), visible=False)
            plt.setp(axes[n, 0].get_yticklabels(), visible=False)
            axes[n, 0].tick_params(axis='both', which='both', length=0)
            img = cv2.imread(img)[:, :, 0]

            axes[n, 0].imshow(img, cmap='jet', vmin=0, vmax=255)
            if n == 0:
                axes[n, 0].set_title('Images', fontsize=10, fontstyle='italic')
            n += 1
    n = 0
    for i, msk in enumerate(org_mask_path):
        if (i + 1) in image_list:
            plt.setp(axes[n, 1].get_xticklabels(), visible=False)
            plt.setp(axes[n, 1].get_yticklabels(), visible=False)
            axes[n, 1].tick_params(axis='both', which='both', length=0)
            msk = read_image(msk)[0, :, :].unsqueeze(0)
            _, h, w = msk.shape

            axes[n, 1].imshow(msk.squeeze(), cmap='jet', vmin=0, vmax=1)
            if n == 0:
                axes[n, 1].set_title('GT', fontsize=10, fontstyle='italic')
            n += 1
    n = 0
    ax_im = []
    numerates = [r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$', r'$(f)$']
    for m, sub_path in enumerate(sub_folders):
        folder = glob.glob(path + '/' + sub_path)
        image_path = glob.glob(folder[0] + '/preds' + '/*')
        n = 0
        for i, img_path in enumerate(image_path):
            if (i + 1) in image_list:
                plt.setp(axes[n, m + 2].get_xticklabels(), visible=False)
                plt.setp(axes[n, m + 2].get_yticklabels(), visible=False)
                axes[n, m + 2].tick_params(axis='both', which='both', length=0)
                img = read_image(img_path)[:, :, :] / 255
                img = torch.where(img >= 0.5, 1, 0)

                ax = axes[n, m + 2].imshow(img[0, :, :], cmap='jet', vmin=0, vmax=1, snap=True)
                if n == 0:
                    axes[n, m + 2].set_title(name_list[m], fontsize=8, fontstyle='normal')
                n += 1
                ax_im.append(ax)
                if n == 5:
                    axes[n, m + 2].set_xlabel(numerates[m])

    plt.savefig(save_name, dpi=400, bbox_inches='tight')
    plt.show()


def plot_relevancemaps(path, sub_folders, name_list, image_list, save_name, figsize=(8, 8), cmap=None):
    fig, axes = plt.subplots(len(image_list), len(sub_folders) + 1, figsize=figsize,
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.0})
    folder = glob.glob(path + '/' + sub_folders[0])
    org_image_path = glob.glob(folder[0] + '/' + 'images' + '/*')
    org_mask_path = glob.glob(folder[0] + '/' + 'labels' + '/*')

    n = 0
    for i, img in enumerate(org_image_path):
        if (i + 1) in image_list:
            plt.setp(axes[n, 0].get_xticklabels(), visible=False)
            plt.setp(axes[n, 0].get_yticklabels(), visible=False)
            axes[n, 0].tick_params(axis='both', which='both', length=0)
            img = cv2.imread(img)[:, :, 0]

            axes[n, 0].imshow(img, cmap='jet', vmin=0, vmax=255)
            if n == 0:
                axes[n, 0].set_title('Images', fontsize=10, fontstyle='italic')
            n += 1
    n = 0
    for i, msk in enumerate(org_mask_path):
        if (i + 1) in image_list:
            plt.setp(axes[n, 1].get_xticklabels(), visible=False)
            plt.setp(axes[n, 1].get_yticklabels(), visible=False)
            axes[n, 1].tick_params(axis='both', which='both', length=0)
            msk = read_image(msk)[0, :, :].unsqueeze(0)

            axes[n, 1].imshow(msk.squeeze(), cmap='jet', vmin=0, vmax=1)
            if n == 0:
                axes[n, 1].set_title('GT', fontsize=10, fontstyle='italic')
            n += 1
    n = 0
    ax_im = []
    for m, sub_path in enumerate(sub_folders[1:]):
        folder = glob.glob(path + '/' + sub_path)
        image_path = glob.glob(folder[0] + '/*')
        n = 0
        for i, img_path in enumerate(image_path):
            if (i + 1) in image_list:
                plt.setp(axes[n, m + 2].get_xticklabels(), visible=False)
                plt.setp(axes[n, m + 2].get_yticklabels(), visible=False)
                axes[n, m + 2].tick_params(axis='both', which='both', length=0)
                img = read_image(img_path)[:, :, :] / 255

                ax = axes[n, m + 2].imshow(img.squeeze(), cmap=cmap, vmin=0, vmax=1)
                if n == 0:
                    axes[n, m + 2].set_title(name_list[m], fontsize=8, fontstyle='normal')
                n += 1
                ax_im.append(ax)

    plt.savefig(save_name, dpi=400, bbox_inches='tight')
    plt.show()


def plot_relevancemaps_2(path, sub_folders, name_list, image_list, save_name, figsize=(8, 8), cmap=None):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    h_ratio = [1, .02]
    outer_grid = gridspec.GridSpec(2, 1, figure=fig, height_ratios=h_ratio)

    inner_grid1 = gridspec.GridSpecFromSubplotSpec(nrows=len(image_list), ncols=(len(sub_folders) + 1),
                                                   subplot_spec=outer_grid[0], hspace=0.05, wspace=0.05)
    print(inner_grid1)
    folder = glob.glob(path + '/' + sub_folders[0])
    org_image_path = glob.glob(folder[0] + '/' + 'images' + '/*')
    org_mask_path = glob.glob(folder[0] + '/' + 'labels' + '/*')

    n = 0
    for i, img in enumerate(org_image_path):
        if (i + 1) in image_list:
            img = cv2.imread(img)[:, :, 0]
            ax = fig.add_subplot(inner_grid1[n, 0])
            ax.imshow(img, cmap='jet', vmin=0, vmax=255)
            ax.tick_params(left=False, right=False, labelleft=False,
                           labelbottom=False, bottom=False, top=False, labeltop=False)
            if n == 0:
                ax.set_title('Images', fontsize=10, fontstyle='italic')
            n += 1
    n = 0
    for i, msk in enumerate(org_mask_path):
        if (i + 1) in image_list:

            msk = read_image(msk)[0, :, :].unsqueeze(0)
            ax = fig.add_subplot(inner_grid1[n, 1])
            ax.imshow(msk.squeeze(), cmap='jet', vmin=0, vmax=1)
            ax.tick_params(left=False, right=False, labelleft=False,
                           labelbottom=False, bottom=False, top=False, labeltop=False)
            if n == 0:
                ax.set_title('GT', fontsize=10, fontstyle='italic')
            n += 1
    n = 0
    ax_im = []
    for m, sub_path in enumerate(sub_folders[1:]):
        folder = glob.glob(path + '/' + sub_path)
        image_path = glob.glob(folder[0] + '/*')
        n = 0
        for i, img_path in enumerate(image_path):
            if (i + 1) in image_list:

                img = read_image(img_path)[:, :, :] / 255
                ax = fig.add_subplot(inner_grid1[n, m + 2])
                im = ax.imshow(img.squeeze(), cmap=cmap, vmin=0, vmax=1)
                ax.tick_params(left=False, right=False, labelleft=False,
                               labelbottom=False, bottom=False, top=False, labeltop=False)
                if n == 0:
                    ax.set_title(name_list[m], fontsize=8, fontstyle='normal')
                n += 1

    inner_grid2 = gridspec.GridSpecFromSubplotSpec(ncols=5, nrows=1, subplot_spec=outer_grid[1])

    ax = fig.add_subplot(inner_grid2[0, 2:])
    cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, orientation='horizontal')

    plt.savefig(save_name, dpi=250, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    from zennit.image import get_cmap, palette
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    folder_path = '../results/images'

    cmap = get_cmap('coldnhot')
    new_colorspace = np.array(palette(cmap, 1))

    y = np.ones((new_colorspace.shape[0], 1))
    new_colorspace = np.concatenate((new_colorspace/255, y), axis=1)
    new_colorspace = ListedColormap(new_colorspace)
    """
    path_list = ['Unet64pluss-randominit-DiceBCELoss', 'relevance_map/unetplusspluss32-randominit',
                 'relevance_map/unetplusspluss32-pretrained', 'relevance_map/unetplusspluss64-randominit',
                 'relevance_map/unetplusspluss64-pretrained']
    name_list = [r'UNet++$^{32}$ -' + '\n RandomInit', 'UNet++$^{32}$ - ' + '\n Pretrained',
                 r'UNet++$^{64}$ - ' + '\n RandomInit', r'UNet++$^{64}$ - ' + '\n Pretrained']

    image_list = [3, 4, 6, 10, 9, 12]
    plot_relevancemaps_2(folder_path, path_list, name_list, image_list, save_name='LPR', figsize=(6, 8), cmap=new_colorspace)
    """
    """
    folder_path = '../results/images'
    image_list_hard = [2, 9, 7, 15, 11, 14]
    image_list_easy = [3, 4, 6, 10, 16, 12]

    path_list = ['Unet64pluss-randominit-DiceBCELoss', 'Unet64pluss-randominit-objaug', 'Unet64pluss-randominit-objaug-imaug']
    name_list = ['Original\n Dataset', 'Generated\n Dataset', 'Generated\n Dataset\n w\ Combined ']
    plot_predictions(folder_path, path_list, name_list, image_list_hard, save_name='Bad-objaug', figsize=(6, 8))
    plot_predictions(folder_path, path_list, name_list, image_list_easy, save_name='Good-objaug', figsize=(6, 8))
    """

    """
    # Initial experiment models - selection of models
    path_list = ['Unet64-randominit', 'Unet64-pretrained', 'Unetpluss64-randominit', 'Unetpluss64-pretrained']

    name_list = [r'UNet$^{64}$' + '\nRandomInit', r'UNet$^{64}$' + '\nPretrained',
                 r'UNet++$^{64}$' + '\nRandomInit', r'UNet++$^{64}$' + '\nPretrained']

    #image_list = [2, 9, 7, 15, 11, 14]
    image_list = [2, 9, 7, 15]
    plot_predictions(folder_path, path_list, name_list, image_list, save_name='Bad-initial-selection', figsize=(6, 5))

    #image_list = [3, 4, 6, 10, 16, 12]
    image_list = [6, 10, 16, 12]
    plot_predictions(folder_path, path_list, name_list, image_list, save_name='Good-initial-selection', figsize=(6, 5))
    """

    # Loss function experiments display
    # Dice, BCE and Focal loss
    image_list_hard = [2, 9, 7, 15]
    image_list_easy = [6, 10, 16, 12]
    """
    path_list = ['Unetpluss64-diceloss',
                 'Unetpluss64-bce',
                 'Unetpluss64-focal'
                 ]

    name_list = [r'$\mathcal{L}_{Dice}$', r'$\mathcal{L}_{BCE}$', r'$\mathcal{L}_{Focal}$']
    plot_predictions(folder_path, path_list, name_list, image_list_hard, save_name='Hard-initial_loss-first', figsize=(6, 8))
    plot_predictions(folder_path, path_list, name_list, image_list_easy, save_name='Easy-initial_loss-first', figsize=(6, 8))

    # Dice-BCE and Dice-Boundary (increase and rebalance approach)
    path_list = [
        'Unetpluss64-dicebce',
        'Unetpluss64-diceboundary-inc',
        'Unetpluss64-diceboundary-reb'
    ]

    name_list = [r'$\mathcal{L}_{Dice} + 'r'\mathcal{L}_{BCE}$',
                 r'$\mathcal{L}_{Dice} + \mathcal{L}_{B}$' + '\n- Increase',
                 r'$\mathcal{L}_{Dice} + \mathcal{L}_{B}$' + '\n- Rebalance']

    plot_predictions(folder_path, path_list, name_list, image_list_hard, save_name='Hard-initial_loss-second', figsize=(6, 8))
    plot_predictions(folder_path, path_list, name_list, image_list_easy, save_name='Easy-initial_loss-second', figsize=(6, 8))
    """
    """
    path_list = ['Unetpluss64-dicebce',
                 'Unetpluss64-orgdata-aug',
                 'Unetpluss64-gendata',
                 'Unetpluss64-gendata-aug'
                 ]
    name_list = ['No Aug', 'Image-Aug',
                 'ObjAug', 'ObjAug\n+ Image-Aug']

    plot_predictions(folder_path, path_list, name_list, image_list_hard, save_name='Hard-dataset', figsize=(6, 5))
    plot_predictions(folder_path, path_list, name_list, image_list_easy, save_name='Easy-dataset', figsize=(6, 5))
    """
    """
    path_list = ['Unetpluss64-diceloss',
                 'Unetpluss64-bce',
                 'Unetpluss64-dicebce',
                 'Unetpluss64-diceboundary-inc',
                 'Unetpluss64-diceboundary-reb'
                 ]
    name_list = [r'$\mathcal{L}_{Dice}$',
                 r'$\mathcal{L}_{BCE}$',
                 r'$\mathcal{L}_{Dice} + 'r'\mathcal{L}_{BCE}$',
                 r'$\mathcal{L}_{Dice} + \mathcal{L}_{B}$' + '\n- Increase',
                 r'$\mathcal{L}_{Dice} + \mathcal{L}_{B}$' + '\n- Rebalance'
                 ]

    plot_predictions(folder_path, path_list, name_list, image_list_hard, save_name='Hard-loss', figsize=(6, 4))
    plot_predictions(folder_path, path_list, name_list, image_list_easy, save_name='Easy-loss', figsize=(6, 4))
    """

    path_list = ['Unetpluss64-dicebce', 'relevance_map/unetpluss64-dice-noaug',
                 'relevance_map/unetpluss64-dicebce-noaug', 'relevance_map/unetpluss64-orgdata-aug',
                 'relevance_map/unetpluss64-gendata-aug']
    name_list = ['NoAug\n' + r'$(\mathcal{L}_{Dice})$', 'NoAug\n' + r'$(\mathcal{L}_{Dice} + \mathcal{L}_{BCE})$',
                 'ImAug\n' + r'$(\mathcal{L}_{Dice} + \mathcal{L}_{BCE})$',
                 'ObjAug + ImAug\n' + r'$(\mathcal{L}_{Dice} + \mathcal{L}_{BCE})$']

    image_list = [4, 10, 9]
    plot_relevancemaps_2(folder_path, path_list, name_list, image_list, save_name='LPR_aug', figsize=(6, 4), cmap=new_colorspace)

    path_list = ['Unetpluss64-dicebce', 'relevance_map/unet32-pretrained',
                 'relevance_map/unet64-pretrained', 'relevance_map/unetpluss32-pretrained',
                 'relevance_map/unetpluss64-pretrained']
    name_list = [r'UNet$^{32}$', r'UNet$^{64}$', r'UNet++$^{32}$', r'UNet++$^{64}$']
    #image_list = [3, 4, 6, 10, 9, 12]
    plot_relevancemaps_2(folder_path, path_list, name_list, image_list, save_name='LPr-pretrained', figsize=(6, 4),
                         cmap=new_colorspace)

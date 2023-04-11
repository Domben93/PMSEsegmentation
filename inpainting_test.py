import math
import random
from typing import Tuple

from torch import Tensor

from partial_conv.src.model import PConvUNet
from PIL import Image
import argparse
import torch
import torch.nn.functional as f
import torchvision.transforms.functional as tf
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from utils.dataset import PMSE_Dataset
from utils.transforms import *
import os
from skimage import measure


def padding(mask: Tensor, value: int = 2):
    mask_copy = torch.zeros(mask.shape)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] == 1:
                mask_copy[max(x - value, 0):min(mask.shape[0], value + x),
                max(y - value, 0):min(mask.shape[1], value + y)] = 1
    return mask_copy


def object_augmentation(obj_image, obj_mask, transforms) -> Tuple[Tensor, Tensor]:

    if transforms:
        for transform in transforms:
            obj_image, obj_mask = transform(obj_image, obj_mask)

    return obj_image, obj_mask


def get_single_PMSE(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Define the model
    print("Loading the Model...")
    model = PConvUNet(finetune=False, layer_size=5, in_ch=1)
    model.load_state_dict(torch.load(args.model, map_location=device)['model'])
    model.to(device)
    model.eval()

    scale_up = QuasiResize([64, 64], 2)
    scale_down = UndoQuasiResize(scale_up)

    dataset = PMSE_Dataset(os.path.join(args.data_path, 'data'), os.path.join(args.data_path, 'label'))

    img, msk, info = dataset[13]

    num = 1
    specific_pmse = 1
    c, h, w = img.shape

    img = img[:, :, :h * num]
    mask = msk[:, :, :h * num]
    mask_padded = torch.logical_not(padding(mask[0, :, :], 1).repeat(3, 1, 1))
    # mask_padded = torch.logical_not(msk[0, :, :]).repeat(3, 1, 1)[:, :, :h*3]
    img_mask, island_count = measure.label(mask.numpy()[0, :, :], background=0, return_num=True, connectivity=1)

    removed_island = torch.where(torch.from_numpy(img_mask) == specific_pmse + 1, 0, torch.from_numpy(img_mask))
    new_mask = torch.where(torch.from_numpy(img_mask) == specific_pmse + 1, 1, 0)
    removed_island = torch.where(removed_island * mask[0, :, :] >= 1, 1, 0)

    removed_island = torch.logical_not(padding(removed_island, 3)).repeat(3, 1, 1)

    inp = (img[:] / 255) * removed_island

    inp_list = []

    for i in range(0, num):
        inp_ = scale_up(inp[:1, :, h * i:h + (h * i)]).to(torch.float32).to(device).view(1, 1, 64, 64)
        mask_ = scale_up(removed_island[:1, :, h * i:h + (h * i)]).to(torch.float32).to(device).view(1, 1, 64, 64)

        with torch.no_grad():
            raw_out, _ = model(inp_, mask_)

            raw_out = raw_out.to(torch.device('cpu')).squeeze()
            raw_out = raw_out.clamp(0.0, 1.0)
            mask_ = mask_.detach().cpu()
            inp_list.append(scale_down((mask_ * inp_.detach().cpu() + (1 - mask_) * raw_out).squeeze(), (h, h)))

    inp_image = torch.cat(inp_list, dim=2)

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(inp_image[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax[1].imshow(new_mask[:, :], cmap='jet', vmin=0, vmax=1)
    plt.show()

    save_path = '/\\dataset\\Test\\single_test'
    save_image(inp_image.to(torch.float32), os.path.join(save_path, 'data', 'img1.png'))
    save_image(new_mask.to(torch.float32) / 255, os.path.join(save_path, 'label', 'img1.png'))



def predict(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Define the model
    print("Loading the Model...")
    model = PConvUNet(finetune=False, layer_size=5, in_ch=1)
    model.load_state_dict(torch.load(args.model, map_location=device)['model'])
    model.to(device)
    model.eval()

    scale_up = QuasiResize([64, 64], 2)
    scale_down = UndoQuasiResize(scale_up)

    dataset = PMSE_Dataset(os.path.join(args.data_path, 'data'), os.path.join(args.data_path, 'label'))

    img, msk, info = dataset[13]

    num = 3
    c, h, w = img.shape

    img = img[:, :, :h*num]
    mask = msk[:, :, :h*num]
    mask_padded = torch.logical_not(padding(mask[0, :, :], 1).repeat(3, 1, 1))
    #mask_padded = torch.logical_not(msk[0, :, :]).repeat(3, 1, 1)[:, :, :h*3]
    inp = (img[:] / 255) * mask_padded

    inp_list = []

    for i in range(0, num):
        print(i)
        inp_ = scale_up(inp[:1, :, h*i:h + (h*i)]).to(torch.float32).to(device).view(1, 1, 64, 64)
        mask_ = scale_up(mask_padded[:1, :, h*i:h + (h*i)]).to(torch.float32).to(device).view(1, 1, 64, 64)

        with torch.no_grad():
            raw_out, _ = model(inp_, mask_)

            raw_out = raw_out.to(torch.device('cpu')).squeeze()
            raw_out = raw_out.clamp(0.0, 1.0)
            mask_ = mask_.detach().cpu()
            inp_list.append(scale_down((mask_ * inp_.detach().cpu() + (1 - mask_) * raw_out).squeeze(), (h, h)))

    inp_image = torch.cat(inp_list, dim=2)

    inpainted_image = inp_image.clone().detach()

    img_mask, island_count = measure.label(mask.numpy()[0, :, :], background=0, return_num=True, connectivity=1)
    new_mask = torch.zeros(mask.shape)
    print(island_count)
    for i in range(0, island_count):
        if i != 1:
            continue
        pixel_island = torch.where(torch.from_numpy(img_mask) == i + 1)

        w1, h1 = torch.min(pixel_island[1]), torch.min(pixel_island[0])
        w2, h2 = torch.max(pixel_island[1]) + 1, torch.max(pixel_island[0]) + 1

        patch_height, patch_width = h2 - h1, w2 - w1

        im, mk = object_augmentation(img[:, h1:h2, w1:w2]/255, mask[0, h1:h2, w1:w2], [])

        if im.shape[-1] != patch_width or im.shape[-2] != patch_height:

            patch_centre = (torch.div(w1 + w2, 2, rounding_mode='trunc'), torch.div(h1 + h2, 2, rounding_mode='trunc')) #((w1 + w2) // 2), ((h1 + h2) // 2)

            new_patch_width = im.shape[-1]
            new_patch_height = im.shape[-2]

            w_min = patch_centre[0] - math.floor(new_patch_width / 2)
            w_max = patch_centre[0] + math.ceil(new_patch_width / 2)
            h_min = patch_centre[1] - math.floor(new_patch_height / 2)
            h_max = patch_centre[1] + math.ceil(new_patch_height / 2)

            i_w_start = max(0, w_min)
            i_h_start = max(0, h_min)
            i_w_end = min(inpainted_image.shape[-1], w_max)
            i_h_end = min(inpainted_image.shape[-2], h_max)

            if w_min < 0:
                im = im[:, :, -w_min:]
                mk = mk[:, -w_min:]
            if h_min < 0:
                im = im[:, -h_min:, :]
                mk = mk[-h_min:, :]
            if w_max > inpainted_image.shape[-1]:
                im = im[:, :, :mk.shape[-1] - (w_max - i_w_end)]
                mk = mk[:, :mk.shape[-1] - (w_max - i_w_end)]
            if h_max > inpainted_image.shape[-2]:
                im = im[:, :mk.shape[-2] - (h_max - i_h_end), :]
                mk = mk[:mk.shape[-2] - (h_max - i_h_end), :]

            inpainted_image[:, i_h_start:i_h_end, i_w_start:i_w_end] = torch.where(mk[: ,:] == 1,
                                                                           im[:, :, :],
                                                                          inpainted_image[:, i_h_start:i_h_end, i_w_start:i_w_end])
            new_mask[:, i_h_start:i_h_end, i_w_start:i_w_end] = mk[:, :]

        else:
            inpainted_image[:, h1:h2, w1:w2] = torch.where(mk[:, :] == 1, im[0, :, :], inpainted_image[0, h1:h2, w1:w2])
            new_mask[:, h1:h2, w1:w2] = mk[:, :]

    fig, ax = plt.subplots(5, 1)
    ax[0].imshow(img[0, :, :]/255, cmap='jet', vmin=0, vmax=1)
    ax[0].axis('off')
    ax[1].imshow(mask[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax[1].axis('off')
    ax[2].imshow(inp_image[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax[2].axis('off')
    ax[3].imshow(new_mask[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax[3].axis('off')
    ax[4].imshow(inpainted_image[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax[4].axis('off')
    plt.show()

    save_path = '/\\dataset\\Test\\single_test'

    save_image(inpainted_image[0, :, :].repeat(3, 1, 1), os.path.join(save_path, 'data', 'img1.png'))
    save_image(new_mask[:, :, :], os.path.join(save_path, 'label', 'img1.png'))

    """
    save_path = 'C:\\Users\\dombe\\Documents\\LaTeX-files\\MasterThesis\\figures'
    fig, ax = plt.subplots()
    ax.imshow(img[0, :, :]/255, cmap='jet', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'original_image.png'),bbox_inches='tight', pad_inches=0)
    fig, ax = plt.subplots()
    ax.imshow(mask[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'original_mask.png'),bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    ax.imshow(img_mask, cmap='jet', vmin=0, vmax=island_count + 1)
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'original_island_mask.png'),bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    ax.imshow(inp_image[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'inpainted_image.png'),bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    ax.imshow(new_mask[0, :, :], cmap='jet', vmin=0, vmax=island_count + 1)
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'new_mask.png'),bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    ax.imshow(inpainted_image[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'new_image.png'),bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    ax.imshow(inp[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'holed_out.png'), bbox_inches='tight', pad_inches=0)
    """


if __name__ == '__main__':
    random.seed(42)
    parser = argparse.ArgumentParser(description="Specify the inputs")
    parser.add_argument('--model', type=str,
                        default="C:\\Users\\dombe\\PycharmProjects\\Test\\partial_conv\\ckpt\\0310_1433_18\\models\\11000.pth")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--data_path', type=str, default="C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Test")
    args = parser.parse_args()

    predict(args)
    #get_single_PMSE(args)

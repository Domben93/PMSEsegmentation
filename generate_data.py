import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision.utils import save_image
from utils.transforms import *
from inpainting.src.model import PConvUNet
from utils.dataset import PMSE_Dataset
import os
import math
from skimage import measure
from tqdm import tqdm
import argparse


class GenerateTrainData:

    def __init__(self, original_data_path,
                 transform: List[FunctionalTransform],
                 model_weight_path: str,
                 device: int,
                 save_path: str,
                 object_mask_padding: int = 1,
                 mask_extraction_padding: int = 0,
                 pixel_island_connectivity: int = 1,
                 random_erase_object: float = .0):
        """

        Args:
            original_data_path: path to data folder, imges must be in a 'image' folder and masks is 'label' folder
            transform: object transforms
            model_weight_path: trained inpainting model weights'
            device: device that the inpainting model is put
            save_path: Path to save the generated data
            object_mask_padding: The number of points outside the objects boundaries that are removed for inpainting
            mask_extraction_padding: The number of points outside the object label tto extract from the image. NOTE:
            setting this to a non-zero value will make the process very slow as the padding process is iterative.
            The padding process needs to be improved!
            pixel_island_connectivity: Connectivity number to consider an object separate from another
        """
        self.data_path = original_data_path
        self.transform = transform
        self.save_path = save_path
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        self.model = PConvUNet(finetune=False, layer_size=5, in_ch=1)
        self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device)['model'])
        self.model.to(device)
        self.model.eval()
        self.random_erase_object = random_erase_object
        self.object_mask_padding = object_mask_padding
        self.mask_extraction_padding = mask_extraction_padding
        self.connectivity = pixel_island_connectivity
        self.scale_up = QuasiResize([64, 64], 2)
        self.scale_down = UndoQuasiResize(self.scale_up)

    def generate(self, number_of_generation_iter, copies_original: int = 1):

        dataset = PMSE_Dataset(os.path.join(self.data_path, 'data'),
                               os.path.join(self.data_path, 'label'),
                               disable_warning=True)

        for num in tqdm(range(number_of_generation_iter), desc='Generating training data'):
            for i in range(len(dataset)):
                o_image, o_mask, info = dataset[i]

                if num < copies_original:
                    save_image(o_image.to(torch.float32) / 255,
                               os.path.join(self.save_path, 'data', info['image_name'] + f'_{str(num)}' + '.png'))
                    save_image(o_mask.to(torch.float32) / 255,
                               os.path.join(self.save_path, 'label', info['image_name'] + f'_{str(num)}' + '.png'))

                c, h, w = o_image.shape
                batch = math.ceil(w / h)

                # o_image = o_image[:, :, :]
                # o_mask = o_mask[:, :, :]

                if self.object_mask_padding:
                    mask = torch.logical_not(self.padding(o_mask[0, :, :], self.object_mask_padding).repeat(3, 1, 1))
                else:
                    mask = torch.logical_not(o_mask.repeat(3, 1, 1))

                inp_list = []
                inp = (o_image / 255) * mask

                for n in range(batch):
                    if n == batch - 1:
                        idx_min = h + (h * (n - 1))
                        idx_max = w
                    else:
                        idx_min = h * n
                        idx_max = h + (h * n)

                    inp_ = self.scale_up(inp[:1, :, idx_min:idx_max]).to(torch.float32). \
                        to(self.device).view(1, 1, 64, 64)
                    mask_ = self.scale_up(mask[:1, :, idx_min:idx_max]).to(torch.float32). \
                        to(self.device).view(1, 1, 64, 64)

                    with torch.no_grad():
                        raw_out, _ = self.model(inp_, mask_)

                        raw_out = raw_out.to(torch.device('cpu')).squeeze()
                        raw_out = raw_out.clamp(0.0, 1.0)
                        mask_ = mask_.detach().cpu()
                        inp_ = inp_.detach().cpu()
                        scaled_down = self.scale_down((mask_ * inp_ + (1 - mask_) * raw_out).squeeze(),
                                                      (h, (idx_max - idx_min)))
                        inp_list.append(scaled_down)

                if len(inp_list) > 1:
                    inp_image = torch.cat(inp_list, dim=2)
                else:
                    inp_image = inp_list[0]

                inpainted_image = inp_image.clone().detach()

                img_mask, island_count = measure.label(o_mask.numpy()[0, :, :], background=0, return_num=True,
                                                       connectivity=self.connectivity)
                new_mask = torch.zeros(o_mask.shape)

                for idx in range(island_count):

                    if self.random_erase_object:
                        if torch.rand(1) < self.random_erase_object:
                            continue
                    # only get the pixels that correspond to the pixel island from the patch
                    # Not that this makes the process very slow
                    island_mask = o_mask * torch.where(torch.from_numpy(img_mask) == idx + 1, 1, 0)
                    if self.mask_extraction_padding:
                        island_mask = self.padding(island_mask[0, :, :], self.mask_extraction_padding)
                    else:
                        island_mask = island_mask[0, :, :]
                    pixel_island = torch.where(island_mask == 1)

                    # get patch bounding box
                    w1, h1 = torch.min(pixel_island[1]), torch.min(pixel_island[0])
                    w2, h2 = torch.max(pixel_island[1]) + 1, torch.max(pixel_island[0]) + 1

                    patch_height, patch_width = h2 - h1, w2 - w1

                    im, mk = self.object_augmentation((o_image[:1, h1:h2, w1:w2] / 255) * island_mask[h1:h2, w1:w2],
                                                      island_mask[h1:h2, w1:w2], self.transform)

                    # unsqueeze tensors which are returned zero -or one-dimensional from object augmentation.
                    if len(mk.shape) == 0:
                        mk = mk.unsqueeze(0).unsqueeze(1)
                    elif len(mk.shape) == 1:
                        if w2 - w1 == 1:
                            mk = mk.unsqueeze(1)
                        else:
                            mk = mk.unsqueeze(0)
                    elif len(mk.shape) == 2:
                        mk = mk.unsqueeze(0)
                    # if size has changed due to expansion of tensor during augmentation
                    if im.shape[-1] != patch_width or im.shape[-2] != patch_height:

                        patch_centre = (int(torch.div(w1 + w2, 2)),
                                        int(torch.div(h1 + h2, 2)))

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
                            mk = mk[:, :, -w_min:]
                        if h_min < 0:
                            im = im[:, -h_min:, :]
                            mk = mk[:, -h_min:, :]
                        if w_max > inpainted_image.shape[-1]:
                            im = im[:, :, :im.shape[-1] - (w_max - i_w_end)]
                            mk = mk[:, :, :mk.shape[-1] - (w_max - i_w_end)]
                        if h_max > inpainted_image.shape[-2]:
                            im = im[:, :im.shape[-2] - (h_max - i_h_end), :]
                            mk = mk[:, :mk.shape[-2] - (h_max - i_h_end), :]

                        inpainted_image[:, i_h_start:i_h_end, i_w_start:i_w_end] = torch.where(mk[:, :, :] == 1,
                                                                                               im[:, :, :],
                                                                                               inpainted_image[:,
                                                                                               i_h_start:i_h_end,
                                                                                               i_w_start:i_w_end])
                        new_mask[:, i_h_start:i_h_end, i_w_start:i_w_end] = mk[:, :] / 255
                    else:

                        inpainted_image[:, h1:h2, w1:w2] = torch.where(mk[:, :] == 1, im[:1, :, :],
                                                                       inpainted_image[:, h1:h2, w1:w2])
                        new_mask[:, h1:h2, w1:w2] = mk[:, :] / 255

                save_image(inpainted_image,
                           os.path.join(self.save_path, 'data', info['image_name'] + f'_{num}' + f'_{i}.png'))
                save_image(new_mask,
                           os.path.join(self.save_path, 'label', info['image_name'] + f'_{num}' + f'_{i}.png'))

    @staticmethod
    def padding(mask: Tensor, value: int = 2):
        mask_copy = torch.zeros(mask.shape)

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x, y] == 1:
                    mask_copy[max(x - value, 0):min(mask.shape[0], value + x),
                    max(y - value, 0):min(mask.shape[1], value + y)] = 1
        return mask_copy

    @staticmethod
    def object_augmentation(obj_image, obj_mask, transforms) -> Tuple[Tensor, Tensor]:

        for transform in transforms:
            obj_image, obj_mask = transform(obj_image, obj_mask)

        return obj_image, obj_mask


class InpaintPadding:

    def __init__(self, data_path: str,
                 save_path: str,
                 device: int,
                 model_weight_path: str,
                 object_mask_padding: int = 1,
                 new_size: Tuple[int] = (64, 64),
                 max_scale: int = 64,
                 continuous_pad: bool = False,
                 continuous_max: int = 10):

        self.data_path = data_path

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            os.mkdir(os.path.join(self.save_path, 'data'))
            os.mkdir(os.path.join(self.save_path, 'label'))

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        self.model = PConvUNet(finetune=False, layer_size=5, in_ch=1)
        self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device)['model'])
        self.model.to(device)
        self.model.eval()

        self.continuous_pad = continuous_pad
        self.continuous_max = continuous_max
        self.object_mask_padding = object_mask_padding
        self.new_size = new_size
        self.scale_up = QuasiResize(list(new_size), max_scale, padding_mode='constant')
        self.scale_up_mask = QuasiResize(list(new_size), max_scale, padding_mode='constant', value=1)
        self.scale_down = UndoQuasiResize(self.scale_up)

    def pad_data(self):

        dataset = PMSE_Dataset(os.path.join(self.data_path, 'data'),
                               os.path.join(self.data_path, 'label'),
                               disable_warning=True)

        for i in range(len(dataset)):
            o_image, o_mask, info = dataset[i]

            c, h, w = o_image.shape
            batch = math.ceil(w / h)

            if self.object_mask_padding:
                mask = torch.logical_not(GenerateTrainData.padding(o_mask[0, :, :],
                                                                   self.object_mask_padding).repeat(3, 1, 1))
            else:
                mask = torch.logical_not(o_mask.repeat(3, 1, 1))

            inp_list, original_im_list, original_mask_list = [], [], []
            inp = (o_image / 255) * mask

            for n in range(batch):
                if n == batch - 1:
                    idx_min = h + (h * (n - 1))
                    idx_max = w
                else:
                    idx_min = h * n
                    idx_max = h + (h * n)

                if idx_max - idx_min < 20:
                    continue

                original_im_list.append(o_image[:, :, idx_min:idx_max] / 255)
                original_mask_list.append(mask[:, :, idx_min:idx_max])

                inp_ = self.scale_up(inp[:1, :, idx_min:idx_max]).to(torch.float32). \
                    to(self.device).view(1, 1, 64, 64)
                mask_ = self.scale_up(mask[:1, :, idx_min:idx_max]).to(torch.float32). \
                    to(self.device).view(1, 1, 64, 64)

                w_continuous = True if (o_image[:, :, idx_min:idx_max].shape[-1] - inp_.shape[-1]) // 2 \
                                       > self.continuous_max else False
                h_continuous = True if (o_image.shape[-2] - inp_.shape[-1]) // 2 > self.continuous_max else False

                if self.continuous_pad and (w_continuous or h_continuous):
                    continuous = True
                    while continuous:
                        break

                else:

                    with torch.no_grad():

                        raw_out, _ = self.model(inp_, mask_)

                        raw_out = raw_out.to(torch.device('cpu')).squeeze()
                        raw_out = raw_out.clamp(0.0, 1.0)
                        mask_ = mask_.detach().cpu()
                        inp_ = inp_.detach().cpu()
                        inpainted_img = (mask_ * inp_ + (1 - mask_) * raw_out).squeeze()

                    inp_list.append(inpainted_img)

            for n, (inp_img, o_img, o_msk) in enumerate(zip(inp_list, original_im_list, original_mask_list)):
                im_h, im_w = inp_img.shape[-2:]
                o_im_h, o_im_w = o_img.shape[-2:]

                w_start = (im_w - o_im_w) // 2
                w_end = w_start + o_im_w
                h_start = (im_h - o_im_h) // 2
                h_end = h_start + o_im_h

                inp_img[h_start:h_end, w_start:w_end] = o_img[0, :, :]
                new_mask = torch.zeros(inp_img.shape[-2:])
                new_mask[h_start:h_end, w_start:w_end] = torch.logical_not(o_msk[0, :, :])

                save_image(inp_img.repeat(3, 1, 1),
                           os.path.join(self.save_path, 'data', info['image_name'] + f'_{n}.png'))
                save_image(new_mask / 255, os.path.join(self.save_path, 'label', info['image_name'] + f'_{n}.png'))


def generate_data(args, transforms):
    data_gen = GenerateTrainData(original_data_path=args.data_path,
                                 transform=transforms,
                                 model_weight_path=args.model_path,
                                 device=args.device,
                                 save_path=args.save_path,
                                 mask_extraction_padding=0,
                                 object_mask_padding=args.padding,
                                 random_erase_object=args.random_erase)

    data_gen.generate(args.num_samples, copies_original=args.num_original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ObjectAug data')
    parser.add_argument('--data-path', type=str, default='../PMSE-segmentation/dataset/Train')
    parser.add_argument('--model-path', type=str,
                        default='../PMSE-segmentation/partial_conv/weights/inpainting_model_weights.pth')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='../PMSE-segementation/dataset/Train/generated_data')
    parser.add_argument('--padding', type=int, default=1, help='Padding range of ground-truth')
    parser.add_argument('--random-erase', type=float, default=.0, help='Percentage of objects to be randomly erased.'
                                                                       'Must be between 0 and 1 where 0 mean none'
                                                                       'erased and 1 means all erased')
    parser.add_argument('--num-samples', type=int, default=50, help='How many times the dataset is run through the'
                                                                    'ObjectAug method.')
    parser.add_argument('--num-original', type=int, default=10, help='Number of copies from the original dataset to'
                                                                     'balance the new dataset with original data.')

    torch.manual_seed(42)
    transforms = [  # RandomHorizontalFlip(0.5),
                    # RandomRotation(p=0.5, rotation_limit=5, rotation_as_max_pixel_shift=True),
                    # RandomResize(p=0.25, scale=[-3, 3]),
                    # RandomPositionShift(p=0.25, max_shift_h=3, max_shift_w=3)
                 ]

    generate_data(parser.parse_args(), transforms)
    """
    gen_dataset = PMSE_Dataset("C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Train\\padded_data\\data",
                               "C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Train\\padded_data\\label")

    org_dataset = PMSE_Dataset("C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Train\\data",
                               "C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Train\\label")

    from matplotlib.patches import Rectangle

    im, msk, info = gen_dataset[0]
    im1, msk1, info1 = gen_dataset[5]
    fig, ax = plt.subplots(2, 2, figsize=(5, 5), dpi=100)

    ax[0, 0].imshow(im[0, :, :], cmap='jet', vmin=0, vmax=255)
    ax[0, 1].imshow(msk[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax[0, 0].add_patch(Rectangle((9, 9), 46, 46, fill=None, alpha=1))
    ax[1, 0].imshow(im1[0, :, :], cmap='jet', vmin=0, vmax=255)
    ax[1, 0].add_patch(Rectangle((21, 21), 22, 22, fill=None, alpha=1))
    ax[1, 1].imshow(msk1[0, :, :], cmap='jet', vmin=0, vmax=1)
    plt.show()
    """

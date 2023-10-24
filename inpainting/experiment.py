import os.path

from comet_ml import Experiment

import torch
from torchvision import transforms
from src.generate_mask import save_generated_masks, save_partial_conv_dataset
from src.dataset import Places2
from src.model import PConvUNet
from src.loss import InpaintingLoss, VGG16FeatureExtractor
from src.train import Trainer
from src.utils import Config, load_ckpt, create_ckpt_dir
import torch.optim as opt

if __name__ == '__main__':
    # set the config
    config = Config("default_config.yml")
    config.ckpt = create_ckpt_dir()
    print("Check Point is '{}'".format(config.ckpt))

    # Define the used device
    device = torch.device("cuda:{}".format(config.cuda_id)
                          if torch.cuda.is_available() else "cpu")

    if config.generate_masks:

        if os.path.exists('../inpainting/data/train/generated_masks') and \
                os.path.exists('../inpainting/data/validation/generated_masks'):

            for f in os.listdir('../inpainting/data/train/generated_masks'):
                os.remove(os.path.join('../inpainting/data/train/generated_masks', f))
            for f in os.listdir('../inpainting/data/validation/generated_masks'):
                os.remove(os.path.join('../inpainting/data/validation/generated_masks', f))
        else:
            os.mkdir('../inpainting/data/train/generated_masks')
            os.mkdir('../inpainting/data/validation/generated_masks')

        save_generated_masks(train_path='../dataset/Train',
                             val_path='../dataset/Validation',
                             num=200,
                             masks_pr_image=config.num_masks,
                             max_sizes=[10, 20])

    # Define the model
    print("Loading the Model...")
    model = PConvUNet(finetune=config.finetune,
                      layer_size=config.layer_size,
                      in_ch=1)
    if config.finetune:
        model.load_state_dict(torch.load(config.finetune)['model'])
    model.to(device)

    # Define the Validation set
    print("Loading the Validation Dataset...")
    dataset_val = Places2('../PMSE-segmentation/dataset/Validation/',
                          config.data_root,
                          data="val")

    # Set the configuration for training
    if config.mode == "train":

        # Define the Places2 Dataset and Data Loader
        print("Loading the Training Dataset...")
        dataset_train = Places2('../PMSE-segmentation/dataset/Train/',
                                config.data_root,
                                data="train")

        # Define the Loss fucntion
        criterion = InpaintingLoss(VGG16FeatureExtractor(),
                                   tv_loss=config.tv_loss).to(device)
        # Define the Optimizer
        lr = config.finetune_lr if config.finetune else config.initial_lr
        if config.optim == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                model.parameters()),
                                         lr=lr,
                                         weight_decay=config.weight_decay)
        elif config.optim == "SGD":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                        lr=lr,
                                        momentum=config.momentum,
                                        weight_decay=config.weight_decay)

        if config.scheduler:
            lr_scheduler = opt.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.step_size,
                                                   gamma=config.gamma,
                                                   last_epoch=config.last_epoch)
        else:
            lr_scheduler = None

        start_iter = 0
        if config.resume:
            print("Loading the trained params and the state of optimizer...")
            start_iter = load_ckpt(config.resume,
                                   [("model", model)],
                                   [("optimizer", optimizer)])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            print("Starting from iter ", start_iter)

        trainer = Trainer(start_iter, config, device, model, dataset_train,
                          dataset_val, criterion, optimizer, experiment=None, schedule=lr_scheduler)

        trainer.iterate()

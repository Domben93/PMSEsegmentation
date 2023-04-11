import torch
from utils.generate_mask import save_generated_masks
from models.unet_plusspluss.unet_plusspluss import Generic_UNetPlusPlus
from models.unets import UNet

if __name__ == '__main__':
    torch.manual_seed(666)
    tr_path = '../Test/partial_conv/data/train'
    val_path = '../Test/partial_conv/data/validation'

    model = Generic_UNetPlusPlus(input_channels=3,
                                 base_num_features=64,
                                 num_classes=1,
                                 num_pool=4,
                                 convolutional_pooling=False,
                                 convolutional_upsampling=True,
                                 deep_supervision=False,
                                 init_encoder=None,
                                 seg_output_use_bias=True,
                                 final_nonlin=None)


    pre_trained = torch.load(
        f'../Test/weights/Unet_plusspluss_64_pretrain-False_loss-DiceBCELoss_optim-adam_generated_dataset_CAdj-Hflip/lr_0.0003_wd_0.006_betas_0.9-0.999_momentum_0.9_freezed-None_0.pt')

    #save_generated_masks(tr_path, val_path, num=200, max_sizes=[10, 20], masks_pr_image=50)
    """
    import torch

    #unet = UNet(3, 1, 32)
    #optim = Adam(unet)

    from utils.transforms import *
    from skimage import measure
    #NUM_MASK = 2
    #dir_name_train = '../Test/partial_conv/data/train/generated_masks'
    #dir_name_validate = '../Test/partial_conv/data/validate/generated_masks'

    #mask_generator = MaskGenerator(rand_seed=42, figs_max_size=[10, 20], pad_range=2)
    data_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Complete\\data'
    label_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Complete\\label'

    mean, std = 9.2158, 1.5720

    min_, max_ = -3 * std, 3 * std

    pair_compose = PairCompose([[Standardize(mean=mean, std=std), MaskClassReduction([0, 1, 2, 3], [0, 1], 0)],
                                [Normalize((0, 255), (min_, max_)), None],
                                [ToGrayscale(output_channels=3), ToGrayscale(output_channels=3)]])

    dataset = PMSE_Dataset(data_path,
                           label_path,
                           transform=pair_compose,
                           disable_warning=True)

    total_objects = 0

    for i in range(len(dataset)):

        data, mask, info = dataset[i]

        img_mask, island_count = measure.label(mask.numpy()[0, :, :], background=0, return_num=True,
                                               connectivity=1)

        total_objects += island_count

    print(total_objects)
    """
    """
    tr_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\partial_conv\\data\\train'
    val_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\partial_conv\\data\\validation'

    #save_partial_conv_dataset(data_path, label_path, 'C:\\Users\\dombe\\PycharmProjects\\Test\\partial_conv\\data')
    save_generated_masks(tr_path, val_path, 50)
    """
    """
    im, test_mask, _ = dataset[0]
    blackedout_mask = mask_gen.generate_inpainting_mask(torch.squeeze(test_mask), 15, 20)

    print(blackedout_mask.shape)
    fig, ax = plt.subplots(4, 1)

    ax[0].imshow(im[0, :, :], cmap='jet')

    msk = torch.logical_not(blackedout_mask).to(torch.int8)

    ax[1].imshow(msk, cmap='jet', vmin=0, vmax=1)

    ax[2].imshow(torch.squeeze(test_mask) + msk * 2, cmap='jet', vmin=0, vmax=2)

    ax[3].imshow(im[0, :, :] * blackedout_mask, cmap='jet')
    plt.show()
    """
    """
    radar_classes = [[], [], [], []]
    for i in range(len(dataset)):

        im, msk, info = dataset[i]
        print(im.shape)
        unique, count = torch.unique(msk, return_counts=True)
        if im.shape[1] == 22:
            radar_classes[0].append(info['image_name'])
        elif im.shape[1] == 46:
            radar_classes[1].append(info['image_name'])
        elif im.shape[1] == 58:
            radar_classes[2].append(info['image_name'])
        elif im.shape[1] == 60:
            radar_classes[3].append(info['image_name'])

        foreground = count[1]
        background = count[0]
    print(radar_classes)
        #print(f'sample {i + 1}')
        #print(foreground / (foreground + background), foreground, background)
        #print(background / (foreground + background))
    """
    """
    mets_model1 = SegMets()
    mets_model2 = SegMets()

    nums = [0, 0, 0]
    tot_len = [0, 0, 0]
    im_names = [[], [], []]

    for i in range(len(dataset)):
        data, label, info = dataset[i]

        if data.shape[-1] <= 150:
            nums[0] += 1
            tot_len[0] += data.shape[-1]
            im_names[0].append(info['image_name'])

        elif data.shape[-1] <= 500:
            nums[1] += 1
            tot_len[1] += data.shape[-1]
            im_names[1].append(info['image_name'])
        else:
            nums[2] += 1
            tot_len[2] += data.shape[-1]
            im_names[2].append(info['image_name'])

            #print(data.shape, info['image_name'])
    print(nums)
    print(tot_len)
    print(im_names)

    for i in range(len(dataset)):
        data, label, info = dataset[i]

        random_estimator = torch.randint(0, 2, label.shape)

        mets_model1(random_estimator, label, info['image_name'])

        random_estimator = torch.randint(0, 3, label.shape)
        random_estimator = torch.where(random_estimator != 1, 0, 1)

        mets_model2(random_estimator, label, info['image_name'])

    print(mets_model1.mIoU(sample_classes=im_names))
    print(mets_model2.mIoU(sample_classes=im_names))

    print('\n')

    print(mets_model1.confusion_matrix(sample_classes=im_names))
    print(mets_model2.confusion_matrix(sample_classes=im_names))

    mod1_fnr_fpr = mets_model1.FPR_FNR(sample_classes=im_names)
    mod2_fnr_fpr = mets_model2.FPR_FNR(sample_classes=im_names)

    print(mod1_fnr_fpr)

    print(cev(mod2_fnr_fpr, mod1_fnr_fpr))
    print(sde(mod2_fnr_fpr, mod1_fnr_fpr))

    print(cev(mod1_fnr_fpr, mod2_fnr_fpr))
    print(sde(mod1_fnr_fpr, mod2_fnr_fpr))
    
    """
    """
    print(torch.sum(label) / torch.numel(label))

    random_estimator = torch.randint(0, 2, label.shape)

    print(mets.mIoU(random_estimator, label, info))
    print()
    con_mat = binary_confusion_matrix(random_estimator, label)
    print(f'FPR: {con_mat[1, 0] / (con_mat[1, 0] + con_mat[1, 1])}')
    print(f'FNR: {con_mat[0, 1] / (con_mat[0, 1] + con_mat[0, 0])}')
    print('\n')

    #random_estimator = torch.randint(0, 3, label.shape)
    #random_estimator = torch.where(random_estimator != 1, 0, 1)

    """
    """  
    path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\models\\options\\unet_config.ymal'

    transf = PairCompose([
        [ConvertDtype(torch.float32), ConvertDtype(torch.float32)],
        [QuasiResize([64, 64], 2), QuasiResize([64, 64], 2)]
    ])

    loader = get_dataloader(path, transf)
    loader_val = get_dataloader(path, transf, mode='validate')

    dataset = PMSE_Dataset(data_path, label_path, transform=transf,
                           square_split=True,
                           percent_overlap=.0,
                           get_trailing=True,
                           lock_vertical=True)

    im, msk, info = dataset[28]
    print(len(dataset))
    print(info)
    fig, ax = plt.subplots(2, 1)

    ax[0].imshow(im[0, :, :], cmap='jet')
    ax[1].imshow(msk[0, :, :], cmap='jet', vmin=0, vmax=1)
    plt.show()
    """
    """
    import seaborn as sns
    import pandas as pd


    sns.set_theme(style="darkgrid")
    path = '../Test/results/UNet_Train_pretrained_freezeNone_DICE_adam.csv'

    data = pd.read_csv(path)
    print(data)
    # Load an example dataset with long-form data
    #fmri = sns.load_dataset("fmri")
    fig, ax = plt.subplots(2, 1, sharex=True)

    sns.set_theme(style="darkgrid")

    sns.lineplot(x="epoch", y="iou",
                 hue="event",
                 data=data, ax=ax[0])

    sns.lineplot(x="epoch", y="loss",
                 hue="event",
                 data=data, ax=ax[1])

    plt.show()
    """
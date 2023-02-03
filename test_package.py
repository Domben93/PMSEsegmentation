import matplotlib.pyplot as plt

if __name__ == '__main__':

    from utils.transforms import QuasiResize, PairCompose, ConvertDtype, MaskClassReduction
    from utils.dataset import get_dataloader, PMSE_Dataset
    import torch
    from torchmetrics.functional.classification import binary_jaccard_index, binary_accuracy, binary_confusion_matrix
    from utils.model_comparison import sde, cev
    from utils.metrics import SegMets

    data_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Complete\\data'
    label_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Complete\\label'

    dataset = PMSE_Dataset(data_path, label_path, transform=PairCompose([[None, MaskClassReduction([0, 1, 2, 3], [0, 1], 0)]]))

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
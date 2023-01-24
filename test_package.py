import matplotlib.pyplot as plt

if __name__ == '__main__':
    from utils.transforms import QuasiResize, PairCompose, ConvertDtype
    from utils.dataset import get_dataloader, PMSE_Dataset
    import torch
    data_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Validation\\data'
    label_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Validation\\label'

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


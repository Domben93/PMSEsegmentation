import os

class Config:

    complete_data = 'dataset/Complete/data'
    complete_mask = 'dataset/Complete/label'

    train_data = 'dataset/Train/data'
    train_mask = 'dataset/Train/label'

    validation_data = 'dataset/Validation/data'
    validation_mask = 'dataset/Validation/label'

    test_data = 'dataset/Test/data'
    test_mask = 'dataset/Test/label'

    model = 'Unet'  # ['Unet', PSPUnet, ResUnet, ResUnetPlus]
    freezed_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # layers/blocks to freeze. must be between [0, max-model-layers]
    pretrained_model = True  # uses the pretrained model
    channels = 1  # or 3 for rgb (PMSE is only one dimensional backscatter)

    mean = None  # if None, it will be calculated. May take a lot of time for big datasets
    std = None  # if None, it will be calculated. May take a lot of time for big datasets

    resize_size = [64, 64]  # The size of the resized images (square size)
    square_size = True  # if True it will override the resize size if not already square
    max_scale = 2  # maximum
    padding_type = None  # 'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'


    # Hyper parameters

    # Optimaizer (Adam)
    epochs = 120
    learning_rate = 1e-3
    weight_decay = 0.01
    momentum = 0.9  # not used for Adam

    # learning scheduler
    step_size = 30
    gamma = 0.30
    last_epoch = -1 # Give number for last epoch for the



class NetworkSettings:
    class Training:
        # Data root folder
        DATA_LOCATION = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\train'
        # data folder
        DATA = os.path.join(DATA_LOCATION, 'data')
        # Masks folder
        MASKS = os.path.join(DATA_LOCATION, 'label')
        # Size of Training Batch
        BATCH_SIZE = 8
        # Number of workers to prepare and load data
        NUM_WORKERS = 8
        # Save model location
        SAVE_LOC = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\weights'
        #
        IMAGE_SIZE = 64

    class Validation:
        # Data root folder
        DATA_LOCATION = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\validation'
        # data folder
        DATA = os.path.join(DATA_LOCATION, 'data')
        # Masks folder
        MASKS = os.path.join(DATA_LOCATION, 'label')
        # Size of Training Batch
        BATCH_SIZE = 8
        # Number of workers to prepare and load data
        NUM_WORKERS = 8
        # Load weights location
        LOAD_LOC = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\weights'
        #
        IMAGE_SIZE = 64

    class Testing:
        # Data root folder
        DATA_LOCATION = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\test'
        # data folder
        DATA = os.path.join(DATA_LOCATION, 'data')
        # Masks folder
        MASKS = os.path.join(DATA_LOCATION, 'label')
        # Size of Training Batch
        BATCH_SIZE = 8
        # Number of workers to prepare and load data
        NUM_WORKERS = 8
        # Load weights location
        LOAD_LOC = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\weights'
        #
        IMAGE_SIZE = 64

    class CompleteSet:
        DATA = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\all_data'
        MASKS = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\all_masks'

    class SmallSamples:
        DATA = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\class3\\data'
        MASKS = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\class3\\label'

    class IntermediateSamples:
        DATA = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\class2\\data'
        MASKS = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\class2\\label'

    class BigSamples:
        DATA = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\class1\\data'
        MASKS = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\class1\\label'

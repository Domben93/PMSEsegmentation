# Segmentation of PMSE signals

A repository for segmentation of PMSE signal using UNet and UNet++ architecture.

The code originates from the work done in a [Master's thesis](https://munin.uit.no/handle/10037/29272) and in a
published [paper](https://www.mdpi.com/2072-4292/15/17/4291).

Clone repository:

```python
   git clone https://github.com/Domben93/PMSEsegmentation.git
```

Install dependencies:

The required libraries are found in [`/requirements.txt`](./requirements.txt) and can be installed using:
```python
   pip install -r requirements.txt
```

## Inference

Put the data samples into the test folder as shown in the structure below.
The sample data is put in the data folder.
**NOTE** that the data must be either `.png` or `.mat` extension files.

    .
    ├── ...
    ├── Dataset
    │   ├── ...    
    │   ├── Test
    │   │   └── data
    │   └── ...                
    └── ...

###Pretrained weights:

Pretrained weights can be downloaded from a shared [google drive](https://drive.google.com/drive/folders/1Mazjq8j2VBTfO6OowQ1FBfi95MODXgNx?usp=sharing) folder.
Only the best performing weights are made available. See [Train](#Train) on how to train your own weights from scratch. 

###Run inference:
- Quick Run

```python
   python inference.py
```

- Specify image, label, model and model-wights paths. It

```python
   python inference.py --images <image_path> --labels <label_path> --model <model_path> --weights <wight_path>
```
For additional inference specifications please see code.

## Train

To train the model put the training samples into the Train folder and validation samples into the Validation folder.
The sample data is put in the data folder and the labels in the label folder as shown in the structure below.
**NOTE** that the data samples and labels must have the same name and must be either `.png` or `.mat` files.

    .
    ├── ...
    ├── Dataset 
    │   ├── Train
    │   │   ├── data
    │   │   └── label
    │   ├── Validation
    │   │   ├── data
    │   │   └── label
    │   └── ...                
    └── ...

Customize `config/unet_config.yaml` with wanted parameters

**Note** that the augmentation must be selected/changed in the code.

For training run:
```python
   python train.py
```
### Training model with ObjectAug generated data

The code does not support simultaneously generation of new data with the ObjectAug method
and at the same time of train a model. Instead, the data can be generated beforehand using
[`/generate_data.py`](./generate_data.py). If the data-path and save-path is not specified 
the path is set to the [`/Dataset/Train`](/dataset/Train) and [`/Dataset/Train/generated_data`](/dataset/Train/generated_data),
respectively. These paths can easily be changed to the preferred directories. 

The ObjectAug model uses a pretrained model for the [inpainting process](https://arxiv.org/pdf/1804.07723.pdf)
which employs a UNet model with partial convolution. The code for the inpainting model is copied from [tanimutomo](https://github.com/tanimutomo/partialconv.git) and
slightly modified.

To train a new inpainting model from scratch please see [`/inpainting/README.md`](/inpainting/README.md). **NOTE** that in
the code from the inpainting model has been altered from the original git repository and might not function according to what the 
README.md says. 

The weights used in the generation of the data used to train the models in the also be downloaded from a shared [google drive](https://drive.google.com/drive/folders/1Mazjq8j2VBTfO6OowQ1FBfi95MODXgNx?usp=sharing) folder.

To run the [`generate_data.py`](generate_data.py) please see available parser arguments in the code, and run it using:

```python
    python generate_data.py
```

## Test

```python
   python test.py
```

***NOTE***:

The code is not as clean or well documented as I would like and apologize for the mess that it may seem. I will try to update 
the code and documentation in the future but as of now if you have any questions feel free write.

### Reference Codes

- https://github.com/tanimutomo/partialconv
- https://github.com/MrGiovanni/UNetPlusPlus

### Original Thesis/Paper
- Thesis
  - Thesis Title:
    - Segmentation of Polar Mesospheric Summer Echoes using Fully Convolutional Network
  - Author:
    - Domben, Erik Seip
  - Affiliation:
    - UiT The Arctic University of Norway

- Paper
  - Paper Title:
    - Using Deep Learning Methods for Segmenting Polar Mesospheric Summer Echoes
  - Authors:
    - Domben, Erik Seip; Sharma, Puneet; Mann, Ingrid
  - Affiliation:
    - UiT The Arctic University of Norway




from models.unets import *


class InitiateModel:

    models = {'unet': UNet,
              'resunet': ResUNet}

    def __init__(self,
                 model: str='unet',
                 weights=None,
                 freezed_layers='all'):
        self.model = model.lower()
        self.model = InitiateModel.models[self.model]

        if weights is None:
            self.model.init

        self.weights = weights



    def initiate(self):



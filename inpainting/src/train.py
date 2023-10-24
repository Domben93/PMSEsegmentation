import torch
from torch.utils.data import DataLoader

from .utils import save_ckpt, to_items
from .evaluate import evaluate


class Trainer(object):
    def __init__(self, step, config, device, model, dataset_train,
                 dataset_val, criterion, optimizer, experiment, schedule):
        self.stepped = step
        self.config = config
        self.device = device
        self.model = model
        self.dataloader_train = DataLoader(dataset_train,
                                           batch_size=config.batch_size,
                                           shuffle=True)

        self.dataloader_val = DataLoader(dataset_val,
                                         batch_size=config.batch_size,
                                         shuffle=True)
        self.criterion = criterion
        self.optimizer = optimizer
        self.experiment = experiment
        self.scheduler = schedule

    def iterate(self):
        print('Start the training')
        for i in range(self.config.max_iter):
            for step, (input, mask, gt) in enumerate(self.dataloader_train):

                loss_dict = self.train(step+self.stepped, input, mask, gt)
                # report the loss
                if i % self.config.log_interval == 0:
                    self.report(i+self.stepped, loss_dict, 'Train')

                # evaluation
                if (i+self.stepped + 1) % self.config.vis_interval == 0 \
                        or i == 0 or i + self.stepped == 0:
                    loss_dict = self.evaluate()
                    self.report(i + self.stepped, loss_dict, 'Validation')
                # save the model
                if (i+self.stepped + 1) % self.config.save_model_interval == 0 \
                        or (i + 1) == self.config.max_iter:
                    print('Saving the model...')
                    save_ckpt('{}/models/{}.pth'.format(self.config.ckpt,
                                                        i+self.stepped + 1),
                              [('model', self.model)],
                              [('optimizer', self.optimizer)],
                              i+self.stepped + 1)

    def train(self, step, input, mask, gt):
        # set the model to training mode
        self.model.train()

        # send the input tensors to cuda
        input = input.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)

        # model forward
        output, _ = self.model(input, mask)
        loss_dict = self.criterion(input, mask, output, gt)
        loss = 0.0
        for key, val in loss_dict.items():
            coef = getattr(self.config, '{}_coef'.format(key))
            loss += coef * val

        # updates the model's params
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        loss_dict['total'] = loss
        return to_items(loss_dict)

    def evaluate(self):

        self.model.eval()
        for input, mask, gt in self.dataloader_val:

            # send the input tensors to cuda
            input = input.to(self.device)
            mask = mask.to(self.device)
            gt = gt.to(self.device)

            # model forward
            with torch.no_grad():
                output, _ = self.model(input, mask)
                loss_dict = self.criterion(input, mask, output, gt)
                loss = 0.0
                for key, val in loss_dict.items():
                    coef = getattr(self.config, '{}_coef'.format(key))
                    loss += coef * val

            loss_dict['total'] = loss
            break

        return to_items(loss_dict)

    def report(self, step, loss_dict, mode):
        print('[{}] [STEP: {:>6}] | Valid Loss: {:.6f} | Hole Loss: {:.6f}'\
              '| TV Loss: {:.6f} | Perc Loss: {:.6f}'\
              '| Style Loss: {:.6f} | Total Loss: {:.6f}'.format(mode,
                        step, loss_dict['valid'], loss_dict['hole'],
                        loss_dict['tv'], loss_dict['perc'],
                        loss_dict['style'], loss_dict['total']))
        if self.experiment is not None:
            self.experiment.log_metrics(loss_dict, step=step)



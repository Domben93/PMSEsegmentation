import torch
import torch.nn as nn
import collections
from torch import Tensor
from typing import Union, List, Callable, NoReturn, TypeVar, Set, Sequence, Any, Optional, Dict, OrderedDict
from collections import OrderedDict
from tqdm import tqdm

T = TypeVar('T')


class BaseModel(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BaseModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def init_random_weights(self,
                            module: nn.Module,
                            init_func: Callable = nn.init.kaiming_normal_,
                            init_layer: Union[List, Any] = nn.Conv2d,
                            _prefix: Optional[str] = None) -> NoReturn:
        """
        Initate weights of model with the specified initiator function.
        Note that in order for this to work all layers of the model must be in a torch.nn.Sequential.
        If there are other torch.nn.Modules objects in the blocks, the method will locate the sequentials of that
        module and initiate the layers of that sequential with the initiator function
        Args:
            module: the model (torch.nn.Module) to initiate the random values on
            init_func: torch.nn.init.**** initiator function (e.g. xavier_uniform_)
            init_layer: layer type to initiate (e.g. nn.Conv2d)
            _prefix: Do NOT use this argument! this is only for internal use when iterating through nested modules.

        """

        if isinstance(init_layer, list):
            init_layer = tuple(init_layer)

        state_dict_copy = module.state_dict()

        children = module.named_children()

        for child_name, child_module in children:

            if isinstance(module.get_submodule(child_name), nn.Sequential):
                for layer_name, layer_type in child_module.named_children():
                    if isinstance(layer_type, init_layer):
                        if not _prefix:
                            key = child_name + '.' + layer_name + '.weight'
                        else:
                            key = _prefix + '.' + child_name + '.' + layer_name + '.weight'

                        state_dict_copy[key] = init_func(torch.empty(state_dict_copy[key].shape), a=1e-2)

            else:
                if isinstance(module.get_submodule(child_name), nn.Module): # New module found, resetting prefix
                    new_prefix = ''
                else:
                    if not _prefix:
                        new_prefix = child_name
                    else:
                        new_prefix = _prefix + '.' + child_name

                self.init_random_weights(child_module, init_func, init_layer, _prefix=new_prefix)

        module.load_state_dict(state_dict_copy)

    def init_weigths_by_layer(self, old_model: Union[nn.Module, dict],
                              model_block_linkage: Union[List[int], OrderedDict[int, int]] = 'auto',
                              rest_random: bool = True,
                              rest_random_func: Callable = nn.init.kaiming_normal_) -> NoReturn:
        """
        Initiate model with weights by layer. For 'auto' the module buildup (i.e., how the different layer are stacked
        in the nn.Sequential) must be the same as "self" for the blocks/layers that you want to transfer weights.
        For list and dict, indices in the state dict are counted and linked to each other, tensor size must match
        between or error will be raised.
        Args:
            old_model: Old model that holds the weights
            model_block_linkage:
            rest_random:
            rest_random_func:

        """
        if isinstance(old_model, nn.Module):
            old_statedict_copy = old_model.state_dict()
        elif isinstance(old_model, dict):
            old_statedict_copy = old_model
        else:
            raise TypeError('old model must be of type nn.Module or dict (nn.Module.state_dict())')

        new_statedict_copy = self.state_dict()

        if model_block_linkage == 'auto':
            for new_block_name, new_block in self.named_children():
                new_state_dict_params = list(new_block.state_dict().values())
                for old_block_name, old_block in old_model.named_children():
                    old_state_dict_params = list(old_block.state_dict().values())

                    if [tensor.shape for tensor in new_state_dict_params] == [tensor.shape for tensor in
                                                                              old_state_dict_params]:
                        print(f'Found match between new model: {new_block_name} and old model {old_block_name}')
                        break
                    else:
                        print(f'Found mismatch between model in block: {new_block_name} '
                              f'and old model in block: {old_block_name}')

        elif isinstance(model_block_linkage, list):
            new_keys = list(new_statedict_copy)
            old_keys = list(old_statedict_copy)

            for num in tqdm(model_block_linkage, desc='Assigning pretrained weights to model'):
                # print(old_statedict_copy[old_keys[num]])
                new_statedict_copy[new_keys[num]] = old_statedict_copy[old_keys[num]]

        elif isinstance(model_block_linkage, dict):
            raise NotImplementedError('Weight initiation using dict not yet supported.')
            # for key, value in model_block_linkage.items():
            # if key > len(old_blocks) or value > len(new_blocks):
            #    raise ValueError(
            #        'Specified linkage dict has values that exceeds the number of blocks in the models')
            # new_blocks[key].load_state_dict(old_blocks[value].state_dict())
        else:
            raise TypeError(f'model_block_linkage must be either "auto",'
                            f' a list or a dict. Got {type(model_block_linkage)}')

        if rest_random:
            if isinstance(model_block_linkage, list):

                new_keys = list(new_statedict_copy)

                init_list = [item for item in [i for i in range(len(list(self.state_dict())))]
                             if item not in model_block_linkage]
                print([new_keys[i] for i in init_list])
                for num in tqdm(init_list, desc=f'Initiating other layers weights using: {rest_random_func.__name__}'):

                    if isinstance(self.get_submodule(new_keys[num].rsplit('.', 1)[-2]), nn.Conv2d):

                        if len(new_statedict_copy[new_keys[num]].shape) <= 1:
                            continue
                        print(new_statedict_copy[new_keys[num]].shape)
                        print(new_keys[num])
                        new_statedict_copy[new_keys[num]] = rest_random_func(torch.empty(new_statedict_copy[new_keys[num]].shape), a=1e-2)

        self.load_state_dict(new_statedict_copy)

    def init_weights(self, weights: Union[OrderedDict, List[Tensor], Dict, Callable],
                     keys: List[str] = None,
                     channel_reduction: int = 0) -> NoReturn:
        """
        Initiate model weights. NOTE! Only works if the weights are from a model with the same architecture
        (i.e, all layers are the same). Use init_weights_by_layers if you want to load weights from a different
         architecture to another (layers must still have the same [..., M, N] shape as weights in the old architecture)

        Args:
            weights: OrderedDict, List of Tensors or List of Callable that initiates the weights.

            OrderedDict: Load state_dict of another model in its entirety. Requires that the models are of the same
            structure. Note that the keys of the ordered dict does not have to be the same as they are renamed
            to the keys of this model anyway. Note also that weights and biases from e.g. BatchNormalization layers
            will also be given to the new model.

            Tensors: If a list of tensors is passed then each nn.Conv2 layer will take those values in sequence. If keys
            are specified then the corresponding value pair to that key will be set and given the tensor weight.
            The number of weights must either be equal to the number of layers or the number of keys. Also, tensor
            weight dimension must be same the as models or errors will occur.

            Callable: If a  callables is passed then each nn.Conv2 layer will be initiated with that callable.
            In case of specified keys the method will only initiate the weights value pair to that key.

            keys: Models OrderedDict keys. Specify which layers to initiate

        """
        print(f'Initiating pre-trained weights for {type(self).__name__}.')

        new_state_dict = self.state_dict()

        if isinstance(weights, OrderedDict):
            def convert_keys(loaded_state_dict: Dict[str, Any], new_state_dict: Dict[str, Any]) -> Dict[str, Any]:
                for load_key, new_key in zip(loaded_state_dict.keys(), new_state_dict.keys()):
                    new_state_dict[new_key] = loaded_state_dict[load_key]
                return new_state_dict

            new_state_dict = convert_keys(weights, new_state_dict)

        elif isinstance(weights, list):
            if not keys and not len(weights) == len(self.layers):
                raise ValueError(f'When keys are not specified then the number of weights in the list must be equal'
                                 f'to the number of layers.\nWas given {len(weights)} weights but have'
                                 f' {len(self.layers)} layers')

            if keys:
                assert len(keys) == len(weights)
                for key, weight in zip(keys, weights):
                    key += '.weight'
                    new_state_dict[key] = weight
            else:
                for i, layer_key in enumerate(self.layers):
                    layer_key += '.weight'
                    new_state_dict[layer_key] = weights[i]

        elif isinstance(weights, Callable):
            if keys:
                for key in keys:
                    key += '.weight'
                    new_state_dict[key] = weights(torch.empty(self.state_dict()[key].shape))
            else:
                for layer_key in self.layers:
                    layer_key += '.weight'
                    new_state_dict[layer_key] = weights(torch.empty(self.state_dict()[layer_key].shape))

        else:
            raise TypeError(f'Expected OrderedDict, List of Tensors or Callable. Got {type(weights)}')

        self.load_state_dict(new_state_dict)

    def freeze_blocks(self, freeze_blocks: Union[str, List[int]] = 'all',
                      freeze_layer_types: List[T] = 'all',
                      display_freeze_info: bool = True) -> NoReturn:

        if freeze_blocks == 'all':
            freeze_block_list = [x for x in range(self.blocks)]

        elif isinstance(freeze_blocks, list) and all(isinstance(item, int) for item in freeze_blocks):
            freeze_block_list = list(set(freeze_blocks))  # unique and sorted

            assert len(
                freeze_block_list) <= self.blocks, 'The number of blocks to freeze must be lower or equal to the ' \
                                                   'number of blocks i the model.'
            assert max(freeze_block_list) <= self.blocks - 1, f'block values in list cannot be bigger than the size' \
                                                              f' number of blocks in the model: {self.blocks - 1}.'
        else:
            raise TypeError(f'Expected string option "all" or a list of integers to the blocks to freeeze. '
                            f'Got {type(freeze_blocks)}.')

        blocks = self.named_children()

        def freeze_sequential(sequential: nn.Sequential, block_name: str, layer_types: Union[str, List[T]] = 'all'):

            if layer_types == 'all':
                for layer in sequential:
                    layer.requires_grad_(False)
                    if display_freeze_info:
                        print(f'Freezing weigths: block "{block_name} and layer "{type(layer)}" '
                              f'requires_grad set to False')

            else:
                for layer in sequential:
                    if isinstance(layer, tuple(layer_types)):
                        layer.requires_grad_(False)
                        if display_freeze_info:
                            print(f'Freezing weigths: block "{block_name} and layer "{type(layer)}" '
                                  f'requires_grad set to False')

        def parallel_block(block: nn.Module, block_name: str, layer_types: Union[str, List[T]] = 'all'):

            for name, item in list(block.named_children()):
                update_name = block_name + '.' + name
                if isinstance(item, nn.Sequential):
                    freeze_sequential(item, block_name=update_name, layer_types=layer_types)

                elif isinstance(item, nn.ModuleList):
                    parallel_block(item, block_name=update_name, layer_types=layer_types)

                else:
                    raise TypeError(f'Type {type(item)} not supported as parallel block. Module with parallel layers '
                                    f'must be put in either nn.ModuleList or nn.Sequential.')

        for num, (block_name, block_module) in enumerate(blocks):

            if num not in freeze_block_list:
                continue

            if isinstance(block_module, nn.Sequential):
                freeze_sequential(block_module, block_name, freeze_layer_types)
                continue

            elif len(list(block_module.children())) > 1:  # found parallel block
                parallel_block(block_module, block_name, freeze_layer_types)

            else:
                raise Exception('Encountered unknown case when iterating over blocks and layers of the model. Block '
                                'must either be a sequential or a module with children modules.')

    def model_layers(self, block: Union[List[int], int] = None):
        """
        Returns: layer names of the model
        """
        layer_names = {}
        block_list = []
        if block is None:
            block_list = [i for i in range(0, self.blocks)]
        elif isinstance(block, List):
            block_list = list(set(block))
            assert len(block_list) <= self.blocks, f'Number of blocks specified cannot be bigger than the number of ' \
                                                   f'actual blocks in the model. Got {len(block)} but model only has ' \
                                                   f'{self.blocks} blocks'
            assert all(elem in [i for i in range(0, self.blocks)] for elem in block_list), f'Block list values must be ' \
                                                                                           f'between 0 and {self.blocks - 1}'
        elif isinstance(block, int):
            block_list.append(block)
        else:
            raise TypeError(f'Expected type List or int. Got {type(block)}.')

        for num, blocks, in enumerate(list(self.named_children())):
            if num not in block_list:
                continue
            layer_names[str(blocks[0])] = []
            for layer in list(blocks[1].named_children()):
                layer_names[str(blocks[0])].append(str(layer[0]))

        return layer_names

    @property
    def blocks(self) -> int:
        return len(list(self.named_children()))

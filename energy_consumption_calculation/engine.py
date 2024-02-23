'''
Copyright (C) 2022 Guangyao Chen. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
from functools import partial

import numpy as np

import torch
import torch.nn as nn
from progress.bar import Bar as Bar
try:
    from spikingjelly.clock_driven import surrogate, neuron, functional
except:
    from spikingjelly.activation_based import surrogate, neuron, functional

from .ops import CUSTOM_MODULES_MAPPING, MODULES_MAPPING
from .utils import syops_to_string, params_to_string

from timm.utils import *
from timm.utils.metrics import *  # AverageMeter, accuracy

def get_syops_pytorch(model, input_res, dataloader=None,
                      print_per_layer_stat=True,
                      input_constructor=None,
                      ost=sys.stdout,
                      verbose=False, ignore_modules=[],
                      custom_modules_hooks={},
                      output_precision=3,
                      syops_units='GMac',
                      param_units='M'):
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    syops_model = add_syops_counting_methods(model)  # dir(syops_model)
    syops_model.eval()
    syops_model.start_syops_count(ost=ost, verbose=verbose,
                                ignore_list=ignore_modules)

    if dataloader is not None:
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        syops_count = np.array([0.0, 0.0, 0.0, 0.0])
        bar = Bar('Processing', max=len(dataloader))
        batch_idx = 0
        for batch, target in dataloader:
            batch_idx += 1

            torch.cuda.empty_cache()

            batch = batch.float().to(next(syops_model.parameters()).device)  # torch.Size([128, 3, 32, 32])

            with torch.no_grad():
                # calculate acc at the same time, to confirm the checkpoint
                output = syops_model(batch)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

            functional.reset_net(syops_model)

            bar.suffix = '({batch}/{size})'.format(batch=batch_idx, size=len(dataloader))
            # print acc
            if batch_idx==len(dataloader)==1 or batch_idx==len(dataloader) or batch_idx % 100 == 0:
                print('  Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            top1=top1_m, top5=top5_m))
            bar.next()
            # # for debug
            # if batch_idx >= 1: break

        bar.finish()
        syops_count, params_count = syops_model.compute_average_syops_cost()  # 整个网络的操作数加和除以累积的batchsize、总的参数和。未对网络中子模块操作。
        # syops_count += syops_item / len(dataloader)
    else:
        if input_constructor:
            input = input_constructor(input_res)
            _ = syops_model(**input)
        else:
            try:
                batch = torch.ones(()).new_empty((1, *input_res),
                                                dtype=next(syops_model.parameters()).dtype,
                                                device=next(syops_model.parameters()).device)
            except StopIteration:
                batch = torch.ones(()).new_empty((1, *input_res))

            _ = syops_model(batch)

        syops_count, params_count = syops_model.compute_average_syops_cost()

    if print_per_layer_stat:
        print_model_with_syops(
            syops_model,
            syops_count,
            params_count,
            ost=ost,
            syops_units=syops_units,
            param_units=param_units,
            precision=output_precision
        )
    syops_model.stop_syops_count()
    CUSTOM_MODULES_MAPPING = {}

    return syops_count, params_count, syops_model


def accumulate_syops(self):
    if is_supported_instance(self):
        return self.__syops__
    else:
        sum = np.array([0.0, 0.0, 0.0, 0.0])
        for m in self.children():
            sum += m.accumulate_syops()
        return sum


def print_model_with_syops(model, total_syops, total_params, syops_units='GMac',
                           param_units='M', precision=3, ost=sys.stdout):

    for i in range(3):
        if total_syops[i] < 1:
            total_syops[i] = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def syops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_syops_cost = self.accumulate_syops()
        accumulated_syops_cost[0] /= model.__batch_counter__
        accumulated_syops_cost[1] /= model.__batch_counter__
        accumulated_syops_cost[2] /= model.__batch_counter__
        accumulated_syops_cost[3] /= model.__times_counter__

        # store info for later analysis
        self.accumulated_params_num = accumulated_params_num
        self.accumulated_syops_cost = accumulated_syops_cost

        return ', '.join([self.original_extra_repr(),
                          params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          syops_to_string(accumulated_syops_cost[0],
                                          units=syops_units, precision=precision),
                          '{:.3%} oriMACs'.format(accumulated_syops_cost[0] / total_syops[0]),
                          syops_to_string(accumulated_syops_cost[1],
                                          units=syops_units, precision=precision),
                          '{:.3%} ACs'.format(accumulated_syops_cost[1] / total_syops[1]),
                          syops_to_string(accumulated_syops_cost[2],
                                          units=syops_units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_syops_cost[2] / total_syops[2]),
                          '{:.3%} Spike Rate'.format(accumulated_syops_cost[3] / 100.),
                          'SpkStat: {}'.format(self.__spkhistc__)])  # print self.__spkhistc__
                          #self.original_extra_repr()])


    def syops_repr_empty(self):
        return ''

    def add_extra_repr(m):
        m.accumulate_syops = accumulate_syops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        if is_supported_instance(m):
            syops_extra_repr = syops_repr.__get__(m)
        else:
            syops_extra_repr = syops_repr_empty.__get__(m)
        if m.extra_repr != syops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = syops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_syops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_syops_count = start_syops_count.__get__(net_main_module)
    net_main_module.stop_syops_count = stop_syops_count.__get__(net_main_module)
    net_main_module.reset_syops_count = reset_syops_count.__get__(net_main_module)
    net_main_module.compute_average_syops_cost = compute_average_syops_cost.__get__(
                                                    net_main_module)

    net_main_module.reset_syops_count()

    return net_main_module


def compute_average_syops_cost(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Returns current mean syops consumption per image.

    """

    for m in self.modules():
        m.accumulate_syops = accumulate_syops.__get__(m)

    syops_sum = self.accumulate_syops()
    syops_sum = np.array([item / self.__batch_counter__ for item in syops_sum])

    for m in self.modules():
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    params_sum = get_model_parameters_number(self)
    return syops_sum, params_sum


def start_syops_count(self, **kwargs):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean syops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_syops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__syops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__syops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_syops_counter_hook_function, **kwargs))


def stop_syops_count(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Stops computing the mean syops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_syops_counter_hook_function)
    # self.apply(remove_syops_counter_variables)  # keep this for later analyses


def reset_syops_count(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_syops_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size
    module.__times_counter__ += 1


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0
    module.__times_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_syops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__') or hasattr(module, '__params__'):
            print('Warning: variables __syops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' syops can affect your code!')
            module.__syops_backup_syops__ = module.__syops__
            module.__syops_backup_params__ = module.__params__
        module.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
        module.__params__ = get_model_parameters_number(module)
        # add __spkhistc__ for each module (by yult 2023.4.18)
        module.__spkhistc__ = None #np.zeros(20)  # assuming there are no more than 20 spikes for one neuron


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_syops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops_handle__'):
            module.__syops_handle__.remove()
            del module.__syops_handle__


def remove_syops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__'):
            del module.__syops__
            if hasattr(module, '__syops_backup_syops__'):
                module.__syops__ = module.__syops_backup_syops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__syops_backup_params__'):
                module.__params__ = module.__syops_backup_params__
        # remove module.__spkhistc__ after print
        if hasattr(module, '__spkhistc__'):
            del module.__spkhistc__

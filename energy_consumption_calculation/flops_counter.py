'''
Copyright (C) 2022 Guangyao Chen - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys

import torch.nn as nn

from .engine import get_syops_pytorch
from .utils import syops_to_string, params_to_string


ssa_info = {'depth': 8, 'Nheads': 12, 'embSize': 768, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-768

def get_energy_cost(model, ssa_info):
    # calculate energy consumption according to E_mac = 4.6 pJ and E_ac = 0.9 pJ
    print('Calculating energy consumption ...')
    conv_linear_layers_info = []
    Nac = 0
    Nmac = 0
    for name, module in model.named_modules():

        if "conv" in name or "head" in name:
            if 'block' in name:
                name_split = name.split('.', 2)
                name = f'block[{name_split[1]}].{name_split[2]}'
                # name = f'{name_split[0]}[{int(name_split[1])}].{name_split[2]}'
            # print(name)
            accumulated_syops_cost = eval(f'model.{name}.accumulated_syops_cost')
            tinfo = (name, module, accumulated_syops_cost)
            conv_linear_layers_info.append(tinfo)
            if abs(accumulated_syops_cost[3] - 100) < 1e-4:  # fr = 100%
                Nmac += accumulated_syops_cost[2]
            else:
                Nac += accumulated_syops_cost[1]
    print('Info of Conv/Linear layers: ')
    for tinfo in conv_linear_layers_info:
        print(tinfo)

    # calculate ops for SSA
    print('SSA info: \n', ssa_info)
    depth = ssa_info['depth']
    Nheads = ssa_info['Nheads']
    embSize = ssa_info['embSize']
    Tsteps = ssa_info['Tsteps']
    patchSize = ssa_info['patchSize']
    embSize_per_head = int(embSize/Nheads)
    SSA_Nac_base = Tsteps * Nheads * pow(patchSize, 2) * embSize_per_head * embSize_per_head
    qkv_fr = []
    for d in range(depth):
        q_lif_r = eval(f'model.block[{d}].attn.q_lif.accumulated_syops_cost[3]') / 100
        k_lif_r = eval(f'model.block[{d}].attn.k_lif.accumulated_syops_cost[3]') / 100
        v_lif_r = eval(f'model.block[{d}].attn.v_lif.accumulated_syops_cost[3]') / 100
        qkv_fr.append([q_lif_r, k_lif_r, v_lif_r])
        # calculate the number of ACs for Q*K*V matrix computation
        tNac = SSA_Nac_base * (min(k_lif_r, v_lif_r) + q_lif_r)
        Nac += tNac
    print('Firing rate of Q/K/V inputs in each block: ')
    print(qkv_fr)

    # calculate energy consumption according to E_mac = 4.6 pJ (1e-12 J) and E_ac = 0.9 pJ
    Nmac = Nmac / 1e9 # G
    Nac = Nac / 1e9 # G
    E_mac = Nmac * 4.6 # mJ
    E_ac = Nac * 0.9 # mJ
    E_all = E_mac + E_ac
    print(f"Number of operations: {Nmac} G MACs, {Nac} G ACs")
    print(f"Energy consumption: {E_all} mJ")
    return


def get_model_complexity_info(model, input_res, dataloader=None,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}, backend='pytorch',
                              syops_units=None, param_units=None,
                              output_precision=2):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)


    if backend == 'pytorch':
        syops_count, params_count, syops_model = get_syops_pytorch(model, input_res, dataloader,
                                                      print_per_layer_stat,
                                                      input_constructor, ost,
                                                      verbose, ignore_modules,
                                                      custom_modules_hooks,
                                                      output_precision=output_precision,
                                                      syops_units=syops_units,
                                                      param_units=param_units)
        # calculate energy consumption according to E_mac = 4.6 pJ and E_ac = 0.9 pJ
        get_energy_cost(syops_model, ssa_info)
    else:
        raise ValueError('Wrong backend name')

    if as_strings:
        syops_string = syops_to_string(
            syops_count[0],
            units=syops_units,
            precision=output_precision
        )
        ac_syops_string = syops_to_string(
            syops_count[1],
            units=syops_units,
            precision=output_precision
        )
        mac_syops_string = syops_to_string(
            syops_count[2],
            units=syops_units,
            precision=output_precision
        )
        params_string = params_to_string(
            params_count,
            units=param_units,
            precision=output_precision
        )
        return [syops_string, ac_syops_string, mac_syops_string], params_string

    return syops_count, params_count

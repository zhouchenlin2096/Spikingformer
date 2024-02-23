'''
Copyright (C) 2022 Guangyao Chen - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import torch
import numpy as np
import torch.nn as nn
try:
    from spikingjelly.clock_driven.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, MultiStepParametricLIFNode, ParametricLIFNode
except:
    from spikingjelly.activation_based.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, MultiStepParametricLIFNode, ParametricLIFNode


def spike_rate(inp):

    # Nspks_max = 30  # 例如for spikformer-8-512, real Nspks_max is 17 (2*8+1=17)；若用此计算Spikformer的能耗，则对应论文Appendix G中计算1）.
    Nspks_max = 1  # 只有真正全0-1矩阵才作为event-driven，计算AC，否则均计算为MAC；若用此计算Spikformer的能耗，则对应论文Appendix G中计算2）.
    num = inp.unique()

    if len(num) <= Nspks_max+1 and inp.max() <= Nspks_max and inp.min() >= 0:
        spkhistc = None

        spike = True
        spike_rate = (inp.sum() / inp.numel()).item()

    else:
        spkhistc = None

        spike = False
        spike_rate = 1

    return spike, spike_rate, spkhistc


def empty_syops_counter_hook(module, input, output):
    module.__syops__ += np.array([0.0, 0.0, 0.0, 0.0])


def upsample_syops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__syops__[0] += int(output_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, _ = spike_rate(output)

    if spike:
        module.__syops__[1] += int(output_elements_count) * rate
    else:
        module.__syops__[2] += int(output_elements_count)

    module.__syops__[3] += rate * 100

def relu_syops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__syops__[0] += int(active_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, _ = spike_rate(output)

    if spike:
        module.__syops__[1] += int(active_elements_count) * rate
    else:
        module.__syops__[2] += int(active_elements_count)

    module.__syops__[3] += rate * 100

def IF_syops_counter_hook(module, input, output):
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, spkhistc = spike_rate(output)
    module.__syops__[1] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def LIF_syops_counter_hook(module, input, output):
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)

    spike, rate, spkhistc = spike_rate(output)
    module.__syops__[1] += int(active_elements_count)
    # module.__syops__[2] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def linear_syops_counter_hook(module, input, output):
    input = input[0]
    spike, rate, spkhistc = spike_rate(input)
    # pytorch checks dimensions, so here we don't care much
    batch_size = input.shape[0]
    output_last_dim = output.shape[-1]
    # bias_syops = output_last_dim if module.bias is not None else 0
    bias_syops = output_last_dim*batch_size if module.bias is not None else 0
    module.__syops__[0] += int(np.prod(input.shape) * output_last_dim + bias_syops)
    if spike:
        module.__syops__[1] += int(np.prod(input.shape) * output_last_dim + bias_syops) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape) * output_last_dim + bias_syops)

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def pool_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 192, 32, 32]) [TB, C, H, W]  # output.shape = torch.Size([4, 192, 16, 16])
    spike, rate, spkhistc = spike_rate(input)
    module.__syops__[0] += int(np.prod(input.shape))

    if spike:
        module.__syops__[1] += int(np.prod(input.shape)) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape))

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def bn_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 48, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)
    batch_syops = np.prod(input.shape)
    if module.affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)

    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def conv_syops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 3, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])  # output.shape = torch.Size([4, 48, 32, 32])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_syops = conv_per_position_syops * active_elements_count

    bias_syops = 0

    if conv_module.bias is not None:

        bias_syops = out_channels * active_elements_count

    overall_syops = overall_conv_syops + bias_syops

    conv_module.__syops__[0] += int(overall_syops)

    if spike:
        conv_module.__syops__[1] += int(overall_syops) * rate
    else:
        conv_module.__syops__[2] += int(overall_syops)

    conv_module.__syops__[3] += rate * 100
    conv_module.__spkhistc__ = spkhistc


def rnn_syops(syops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    syops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    syops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        syops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        syops += rnn_module.hidden_size
        # adding operations from both states
        syops += rnn_module.hidden_size*3
        # last two hadamard product and add
        syops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        syops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return syops


def rnn_syops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison syops should be accurate
    """
    syops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        syops = rnn_syops(syops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    syops *= seq_length
    if rnn_module.bidirectional:
        syops *= 2
    rnn_module.__syops__[0] += int(syops)

def rnn_cell_syops_counter_hook(rnn_cell_module, input, output):
    syops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    syops = rnn_syops(syops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    rnn_cell_module.__syops__[0] += int(syops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    syops = 0

    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    syops = 0

    # Q scaling
    syops += qlen * qdim

    # Initial projections
    syops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        syops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_syops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    syops += num_heads * head_syops

    # final projection, bias is always enabled
    syops += qlen * vdim * (vdim + 1)

    syops *= batch_size
    multihead_attention_module.__syops__[0] += int(syops)


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_syops_counter_hook,
    nn.Conv2d: conv_syops_counter_hook,
    nn.Conv3d: conv_syops_counter_hook,
    # activations
    nn.ReLU: relu_syops_counter_hook,
    nn.PReLU: relu_syops_counter_hook,
    nn.ELU: relu_syops_counter_hook,
    nn.LeakyReLU: relu_syops_counter_hook,
    nn.ReLU6: relu_syops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_syops_counter_hook,
    nn.AvgPool1d: pool_syops_counter_hook,
    nn.AvgPool2d: pool_syops_counter_hook,
    nn.MaxPool2d: pool_syops_counter_hook,
    nn.MaxPool3d: pool_syops_counter_hook,
    nn.AvgPool3d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_syops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_syops_counter_hook,
    nn.BatchNorm2d: bn_syops_counter_hook,
    nn.BatchNorm3d: bn_syops_counter_hook,

    # Neuron IF
    MultiStepIFNode: IF_syops_counter_hook,
    IFNode: IF_syops_counter_hook,
    # Neuron LIF
    MultiStepLIFNode: LIF_syops_counter_hook,
    LIFNode: LIF_syops_counter_hook,
    # Neuron PLIF
    MultiStepParametricLIFNode: LIF_syops_counter_hook,
    ParametricLIFNode: LIF_syops_counter_hook,

    nn.InstanceNorm1d: bn_syops_counter_hook,
    nn.InstanceNorm2d: bn_syops_counter_hook,
    nn.InstanceNorm3d: bn_syops_counter_hook,
    nn.GroupNorm: bn_syops_counter_hook,
    # FC
    nn.Linear: linear_syops_counter_hook,
    # Upscale
    nn.Upsample: upsample_syops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_syops_counter_hook,
    nn.ConvTranspose2d: conv_syops_counter_hook,
    nn.ConvTranspose3d: conv_syops_counter_hook,
    # RNN
    nn.RNN: rnn_syops_counter_hook,
    nn.GRU: rnn_syops_counter_hook,
    nn.LSTM: rnn_syops_counter_hook,
    nn.RNNCell: rnn_cell_syops_counter_hook,
    nn.LSTMCell: rnn_cell_syops_counter_hook,
    nn.GRUCell: rnn_cell_syops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_syops_counter_hook

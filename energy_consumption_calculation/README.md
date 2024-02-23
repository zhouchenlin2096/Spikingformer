
## Get started quickly：energy consumption calculation on ImageNet
Download the trained model first [here](https://pan.baidu.com/s/1LsECpFOxh30O3vHWow8OGQ), passwords: abcd
```
cd imagenet
python energy_consumption_calculation_on_imagenet.py
```


## Usage for calculating energy consumption of Spikingformer

0. Please refer to [syops-counter](https://github.com/iCGY96/syops-counter) for more information, on which this repo is based
1. Initialize and load pretrained weights for the model
2. Construct the `DataLoader`
3. Execute the following commands (need to modify `ssa_info` in `flops_counter.py` for different network structures)
4. See `test_spikingformer-8-512_eg.log` for exemplar results

    ```python
    print(args)
    ts1 = time.time()
    import energy_consumption_calculation
    from energy_consumption_calculation.flops_counter import get_model_complexity_info
    # using real data
    Nops, Nparams = get_model_complexity_info(model, (3, 224, 224), loader_eval, as_strings=True, print_per_layer_stat=True, verbose=True, syops_units='Mac', param_units=' ', output_precision=3)
    # using random input
    # Nops, Nparams = get_model_complexity_info(model, (3, 224, 224), dataloader=None, as_strings=True, print_per_layer_stat=True, verbose=True, syops_units='Mac', param_units=' ', output_precision=3)
    print("Nops: ", Nops)
    print("Nparams: ", Nparams)
    t_cost = (time.time() - ts1) / 60
    print(f"Time cost: {t_cost} min")

    # need to modify ssa_info in flops_counter.py (Line 16) for different network structures
    # ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 384, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-384
    # ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 512, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-512
    # ssa_info = {'depth': 8, 'Nheads': 12, 'embSize': 768, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-768

    # just run one batch when debugging (Line 79 in engine.py)
    # if batch_idx >= 1: break
    ```

    ```python
    # output example
    SSA info: 
    {'depth': 8, 'Nheads': 8, 'embSize': 512, 'patchSize': 14, 'Tsteps': 4}
    Firing rate of Q/K/V inputs in each block: 
    [[0.17230188630104062, 0.03626344770312309, 0.06957707978010177], [0.11627475060462952, 0.040295450145006184, 0.07173207724809647], [0.10669307922124864, 0.044414592387676234, 0.07524623577594756], [0.08188644799470902, 0.05000199048876763, 0.0768778057050705], [0.06365273276925086, 0.04325467040181161, 0.07054187965631485], [0.07711239219903945, 0.0311313860142231, 0.07321582739830017], [0.07023756886482238, 0.024202287513613703, 0.07348622434139251], [0.09727698216676713, 0.02812242347121239, 0.07954712931871415]]
    Number of operations: 0.347329512 G MACs, 6.51754094076244 G ACs
    Energy consumption: 7.463502601886196 mJ
    ```

__Comments, issues, contributions, and collaborations are all welcomed!__ (yult0821@163.com)

## Acknowledgements

This repository is developed based on [syops-counter](https://github.com/iCGY96/syops-counter) and [ptflops](https://github.com/sovrasov/flops-counter.pytorch).

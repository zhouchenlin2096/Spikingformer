'''
Copyright (C) 2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''


def syops_to_string(syops, units=None, precision=2):
    if units is None:
        if syops // 10**9 > 0:
            return str(round(syops / 10.**9, precision)) + ' G Ops'
        elif syops // 10**6 > 0:
            return str(round(syops / 10.**6, precision)) + ' M Ops'
        elif syops // 10**3 > 0:
            return str(round(syops / 10.**3, precision)) + ' K Ops'
        else:
            return str(syops) + ' Ops'
    else:
        if units == 'G Ops':
            return str(round(syops / 10.**9, precision)) + ' ' + units
        elif units == 'M Ops':
            return str(round(syops / 10.**6, precision)) + ' ' + units
        elif units == 'K Ops':
            return str(round(syops / 10.**3, precision)) + ' ' + units
        else:
            return str(syops) + ' Ops'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, precision)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, precision)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)

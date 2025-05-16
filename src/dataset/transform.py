import numpy as np


def zero_crossings(voltage):
    """ find the indexes of all points of zero crossings
        找到所有的过零点
    """
    return np.where(np.diff(np.sign(voltage)))[0]


def get_zero_crossing(voltage, NN=1000):
    """ get zero crossing

    """
    zero_crossing = zero_crossings(voltage)

    if len(zero_crossing) > 0:
        if voltage[zero_crossing[0] + 1] > 0:  # 从正的 周波开始
            zero_crossing = zero_crossing[0:]  # 选择所有点
        else:  # 从负的 周波开始
            zero_crossing = zero_crossing[1:]  # 选择第二个点开始

        if len(zero_crossing) % 2 == 1:  # 保证过零点为偶数 （为什么？）   最后一定剩一个半波
            zero_crossing = zero_crossing[:-1]

        if zero_crossing[-1] + NN >= len(voltage):
            zero_crossing = zero_crossing[:-2]
    else:
        zero_crossing = None

    return zero_crossing


def align_IV_zero_crossing(i, v, TS, app):
    """ get whited data (以电压过零点为初始相位取一个周波电压，电流)

        Arguments:
            data = {list: 1339} -- voltage and current data (each element is a {ndarray: (224810, 2)})
            label = {ndarray: (1339,)} -- appliance types

        Returns:
           I = {ndarray: (885,)} - one representative period of current
           V = {ndarray: (885,)} - one representative period of voltage
    """

    ks = []
    cs = []
    current, voltage = np.copy(i), np.copy(v)

    zc = get_zero_crossing(voltage, TS)[1:]
    ks = []
    crs = []

    for j in range(2, len(zc) - 2):
        ts = zc[-j] - zc[-(j + 2)]
        I = current[zc[-(j + 2)]:zc[-j]]
        if app == 'Iron':
            diff = round(np.max(abs(current)), 3) - round(np.max(abs(I)), 3)
            diff = diff * 100 / round(np.max(abs(i)), 3)

        ic = zero_crossings(I)

        if ts >= TS - 100:  # 如果三个过零点的间距大于 Ts - 100
            if len(ic) > 1:  # 且I找到了2个过零点
                k = j
                break

        elif ts > 3 * TS // 2 and ts < TS - 1:
            if len(ic) > 1:
                if app == 'Iron' and diff <= 3:
                    k = j
                    break
                else:
                    k = j
                    break

        elif ts > TS // 2:
            if len(ic) > 1:
                if app == 'Iron' and diff <= 3:
                    k = j
                    break
                else:
                    k = j
                    break
    addition = 0
    if ts != TS:
        addition = TS - ts

    voltage = voltage[zc[-(k + 2)]:zc[-k] + addition]
    current = current[zc[-(k + 2)]:zc[-k] + addition]
    return current, voltage





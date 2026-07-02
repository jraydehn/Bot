"""
pip_features.py

Self-contained Perceptually Important Points (PIP) algorithm with a
compute_pip_shape() wrapper that extracts three shape features from a
close-price array.  No external dependencies beyond numpy.

Shape features returned by compute_pip_shape():
  pip_last_slope : slope of last PIP segment in log-price space (pos=up, neg=down)
  pip_up_frac    : fraction of total PIP amplitude from upward legs [0, 1]
  pip_n_turns    : number of direction reversals in the PIP skeleton [0, n_pips-2]
"""

import numpy as np


def find_pips(data: np.ndarray, n_pips: int, dist_measure: int):
    """
    Iteratively select the n_pips most perceptually important points.

    dist_measure:
        1 = Euclidean  (both axes normalized to [0,1])
        2 = Perpendicular (recommended — geometrically meaningful)
        3 = Vertical

    Returns (pips_x, pips_y) — lists of bar indices and price values.
    """
    pips_x = [0, len(data) - 1]
    pips_y = [float(data[0]), float(data[-1])]

    x_range = len(data) - 1
    y_min, y_max = float(data.min()), float(data.max())
    y_range = y_max - y_min if y_max != y_min else 1.0

    for _ in range(2, n_pips):
        md = 0.0
        mdi = -1
        insert_index = -1

        for k in range(len(pips_x) - 1):
            la, ra = k, k + 1
            time_diff  = pips_x[ra] - pips_x[la]
            price_diff = pips_y[ra] - pips_y[la]
            slope      = price_diff / time_diff
            intercept  = pips_y[la] - slope * pips_x[la]

            for i in range(pips_x[la] + 1, pips_x[ra]):
                d = 0.0
                if dist_measure == 1:
                    nx_i  = i / x_range
                    ny_i  = (data[i] - y_min) / y_range
                    d  = ((pips_x[la] / x_range - nx_i) ** 2 + ((pips_y[la] - y_min) / y_range - ny_i) ** 2) ** 0.5
                    d += ((pips_x[ra] / x_range - nx_i) ** 2 + ((pips_y[ra] - y_min) / y_range - ny_i) ** 2) ** 0.5
                elif dist_measure == 2:
                    d = abs(price_diff * i - time_diff * data[i]
                            + pips_x[ra] * pips_y[la] - pips_y[ra] * pips_x[la])
                    d /= (time_diff ** 2 + price_diff ** 2) ** 0.5
                elif dist_measure == 3:
                    d = abs(data[i] - (slope * i + intercept))

                if d > md:
                    md = d
                    mdi = i
                    insert_index = ra

        pips_x.insert(insert_index, mdi)
        pips_y.insert(insert_index, float(data[mdi]))

    return pips_x, pips_y


def compute_pip_shape(close_arr: np.ndarray, n_pips: int = 5, n_bars: int = 50):
    """
    Run PIP on the last n_bars of close_arr (log-price space) and return
    three shape scalars.  Returns (nan, nan, -1) on failure.

    pip_last_slope : slope of final PIP segment (log-price / bar)
    pip_up_frac    : upward amplitude / total amplitude  [0, 1]
    pip_n_turns    : direction reversals in skeleton     [0, n_pips-2]
    """
    _nan = float("nan")
    try:
        window = close_arr[-n_bars:] if len(close_arr) >= n_bars else close_arr
        if len(window) < n_pips + 2:
            return _nan, _nan, -1
        log_w = np.log(window.astype(float))
        px, py = find_pips(log_w, n_pips, 2)  # perpendicular distance

        # last segment slope
        run = px[-1] - px[-2]
        last_slope = round((py[-1] - py[-2]) / run, 6) if run > 0 else 0.0

        # up / down amplitude
        segs = [py[k + 1] - py[k] for k in range(len(py) - 1)]
        up_amp   = sum(s for s in segs if s > 0)
        down_amp = sum(abs(s) for s in segs if s < 0)
        total_amp = up_amp + down_amp
        up_frac = round(up_amp / total_amp, 4) if total_amp > 0 else 0.5

        # direction changes
        dirs = [1 if s > 0 else -1 for s in segs if s != 0]
        n_turns = sum(1 for k in range(len(dirs) - 1) if dirs[k] != dirs[k + 1])

        return last_slope, up_frac, n_turns
    except Exception:
        return _nan, _nan, -1

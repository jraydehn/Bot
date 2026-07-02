from typing import List

import numpy as np
import pandas as pd

from local_extreme import LocalExtreme


class ATRDirectionalChange:
    def __init__(self, atr_lookback: int):
        self._up_move     = True
        self._pend_max    = np.nan
        self._pend_min    = np.nan
        self._pend_max_i  = 0
        self._pend_min_i  = 0
        self._atr_lb      = atr_lookback
        self._atr_sum     = np.nan
        self._initialized = False
        self.extremes: List[LocalExtreme] = []

    def _create_ext(self, ext_type, ext_i, conf_i, time_index, high, low, close):
        arr = high if ext_type == "high" else low
        self.extremes.append(LocalExtreme(
            ext_type       = 1 if ext_type == "high" else -1,
            index          = ext_i,
            price          = arr[ext_i],
            timestamp      = time_index[ext_i],
            conf_index     = conf_i,
            conf_price     = close[conf_i],
            conf_timestamp = time_index[conf_i],
        ))

    def update(self, i, time_index, high, low, close):
        if i < self._atr_lb:
            return
        elif i == self._atr_lb:
            h_win = high[i - self._atr_lb + 1: i + 1]
            l_win = low [i - self._atr_lb + 1: i + 1]
            c_win = close[i - self._atr_lb: i]
            tr1 = h_win - l_win
            tr2 = np.abs(h_win - c_win)
            tr3 = np.abs(l_win - c_win)
            self._atr_sum = np.sum(np.max(np.stack([tr1, tr2, tr3]), axis=0))
        else:
            tr_curr = max(high[i] - low[i],
                          abs(high[i] - close[i-1]),
                          abs(low[i]  - close[i-1]))
            rm     = i - self._atr_lb
            tr_rm  = max(high[rm] - low[rm],
                         abs(high[rm] - close[rm-1]),
                         abs(low[rm]  - close[rm-1]))
            self._atr_sum += tr_curr - tr_rm

        atr = self._atr_sum / self._atr_lb

        if not self._initialized:
            self._pend_max   = high[i]
            self._pend_min   = low[i]
            self._pend_max_i = self._pend_min_i = i
            self._initialized = True
            return

        if self._up_move:
            if high[i] > self._pend_max:
                self._pend_max   = high[i]
                self._pend_max_i = i
            elif low[i] < self._pend_max - atr:
                self._create_ext("high", self._pend_max_i, i, time_index, high, low, close)
                self._up_move    = False
                self._pend_min   = low[i]
                self._pend_min_i = i
        else:
            if low[i] < self._pend_min:
                self._pend_min   = low[i]
                self._pend_min_i = i
            elif high[i] > self._pend_min + atr:
                self._create_ext("low", self._pend_min_i, i, time_index, high, low, close)
                self._up_move    = True
                self._pend_max   = high[i]
                self._pend_max_i = i

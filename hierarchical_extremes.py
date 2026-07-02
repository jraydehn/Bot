import copy
from typing import List, Optional

import numpy as np
import pandas as pd

from atr_directional_change import ATRDirectionalChange
from local_extreme import LocalExtreme, extremes_sanity_checks


class HierarchicalExtremes:
    def __init__(self, levels: int, atr_lookback: int):
        self._base_dc = ATRDirectionalChange(atr_lookback)
        self._levels  = levels
        self.extremes: List[List[LocalExtreme]] = [[] for _ in range(levels)]

    @staticmethod
    def _comparison(x, y, ext_type: int) -> bool:
        return x > y if ext_type == 1 else x < y

    def _new_ext(self, level, conf_i, conf_price, conf_time, ext_type):
        if level >= self._levels - 1:
            return

        ext_i   = len(self.extremes[level]) - 1
        new_ext = self.extremes[level][ext_i]
        assert new_ext.ext_type == ext_type

        if ext_i < 4:
            return

        prev_ext = self.extremes[level][ext_i - 2]
        assert prev_ext.ext_type == ext_type
        if not self._comparison(prev_ext.price, new_ext.price, ext_type):
            return

        prev_next_lvl = None
        if self.extremes[level + 1]:
            prev_next_lvl = self.extremes[level + 1][-1]
            if prev_next_lvl.ext_type != ext_type:
                if not self._comparison(prev_ext.price, prev_next_lvl.price, ext_type):
                    return

        for prior_i in range(ext_i - 4, -1, -2):
            prior = self.extremes[level][prior_i]
            assert prior.ext_type == ext_type

            if self._comparison(prior.price, prev_ext.price, ext_type):
                return
            if prev_next_lvl is not None and prior.index <= prev_next_lvl.index:
                break
            elif prior.price == prev_ext.price:
                prev_ext = prior
            elif self._comparison(prior.price, prev_ext.price, -ext_type):
                break

        new_ext = copy.copy(prev_ext)
        new_ext.conf_index     = conf_i
        new_ext.conf_price     = conf_price
        new_ext.conf_timestamp = conf_time

        if prev_next_lvl is not None and prev_next_lvl.ext_type == ext_type:
            upgrade_point = None
            for j in range(ext_i - 1, -1, -2):
                prior = self.extremes[level][j]
                assert prior.ext_type == -ext_type
                if prior.index >= new_ext.index:
                    continue
                if prior.index <= prev_next_lvl.index:
                    break
                if upgrade_point is None or not self._comparison(prior.price, upgrade_point.price, ext_type):
                    upgrade_point = prior

            assert upgrade_point is not None
            upgraded               = copy.copy(upgrade_point)
            upgraded.conf_index    = conf_i
            upgraded.conf_price    = conf_price
            upgraded.conf_timestamp = conf_time
            self.extremes[level + 1].append(upgraded)
            self._new_ext(level + 1, conf_i, conf_price, conf_time, -ext_type)

        self.extremes[level + 1].append(new_ext)
        self._new_ext(level + 1, conf_i, conf_price, conf_time, ext_type)

    def update(self, i: int, time_index: pd.DatetimeIndex,
               high: np.ndarray, low: np.ndarray, close: np.ndarray):
        prev_len = len(self._base_dc.extremes)
        self._base_dc.update(i, time_index, high, low, close)
        if len(self._base_dc.extremes) <= prev_len:
            return
        new_ext = self._base_dc.extremes[-1]
        self.extremes[0].append(new_ext)
        self._new_ext(0, i, close[i], time_index[i], new_ext.ext_type)

    def _get_level_extreme(self, level: int, ext_type: int,
                           lag: int = 0) -> Optional[LocalExtreme]:
        if not self.extremes[level]:
            return None
        offset = 0 if self.extremes[level][-1].ext_type == ext_type else 1
        idx = lag * 2 + offset
        if idx >= len(self.extremes[level]):
            return None
        return self.extremes[level][-(idx + 1)]

    def get_level_high(self, level: int, lag: int = 0) -> Optional[LocalExtreme]:
        return self._get_level_extreme(level, 1, lag)

    def get_level_low(self, level: int, lag: int = 0) -> Optional[LocalExtreme]:
        return self._get_level_extreme(level, -1, lag)

    def get_level_high_price(self, level: int, lag: int = 0) -> float:
        ext = self._get_level_extreme(level, 1, lag)
        return ext.price if ext is not None else np.nan

    def get_level_low_price(self, level: int, lag: int = 0) -> float:
        ext = self._get_level_extreme(level, -1, lag)
        return ext.price if ext is not None else np.nan

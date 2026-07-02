from dataclasses import dataclass
import pandas as pd


@dataclass
class LocalExtreme:
    ext_type: int           # +1 = swing high, -1 = swing low
    index: int              # bar index of the extreme
    price: float            # price at the extreme (high or low)
    timestamp: pd.Timestamp
    conf_index: int         # bar index where the swing was confirmed
    conf_price: float       # close at confirmation bar
    conf_timestamp: pd.Timestamp


def extremes_sanity_checks(extremes):
    """Alternation, monotonic index, and price ordering checks."""
    if len(extremes) < 2:
        return

    ext_df = pd.DataFrame([{
        'ext_type': e.ext_type,
        'index':    e.index,
        'price':    e.price,
    } for e in extremes])

    # Always alternating
    assert len(ext_df[ext_df['ext_type'] == ext_df['ext_type'].shift()]) == 0

    # Extreme index is always increasing
    assert ext_df['index'].diff().min() >= 0

    ext_df['last'] = ext_df['price'].shift()

    # All highs are greater than or equal to prior low (equal allowed for flat markets)
    high_exts = ext_df[ext_df['ext_type'] == 1]
    assert len(high_exts[high_exts['price'] < high_exts['last']]) == 0

    # All lows are less than or equal to prior high
    low_exts = ext_df[ext_df['ext_type'] == -1]
    assert len(low_exts[low_exts['price'] > low_exts['last']]) == 0

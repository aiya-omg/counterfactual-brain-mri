"""ADC 比を発症後日数へ換算する。文献の数値は Bon が CSV に入れる。"""

from __future__ import annotations

import pandas as pd

ASSUMPTION = (
    "ADC は発症後に低下してから上昇する。比から日数は一意に決まらないので、"
    "ADC 比が最小になる日以降の上昇期だけを線形補間している。"
)


def rising_phase(table: pd.DataFrame) -> pd.DataFrame:
    """ADC 比が底を打った日以降の行だけを、日数順で返す。"""
    if table.empty or table["adc_ratio"].isna().all():
        raise ValueError("literature_adc_timecourse.csv に値がありません。Bon が文献値を入れるまで換算しない。")
    bottom = table.loc[table["adc_ratio"].idxmin(), "day"]
    rising = table[table["day"] >= bottom].sort_values("day")
    if len(rising) < 2:
        raise ValueError("上昇期の点が2つ未満で、補間できない。")
    return rising


def adc_ratio_to_day(adc_ratio: float, table: pd.DataFrame) -> float:
    """上昇期を仮定した線形補間。範囲外は端点で止めず、エラーにする。"""
    rising = rising_phase(table)
    days = rising["day"].to_numpy(dtype=float)
    ratios = rising["adc_ratio"].to_numpy(dtype=float)
    return _interp(adc_ratio, days, ratios)


def _interp(adc_ratio: float, days, ratios) -> float:
    order = ratios.argsort()
    ratios, days = ratios[order], days[order]
    for left in range(len(ratios) - 1):
        lo, hi = ratios[left], ratios[left + 1]
        if lo <= adc_ratio <= hi:
            if hi == lo:
                return float(days[left])
            weight = (adc_ratio - lo) / (hi - lo)
            return float(days[left] + weight * (days[left + 1] - days[left]))
    raise ValueError(f"ADC 比 {adc_ratio} は上昇期の範囲外。")

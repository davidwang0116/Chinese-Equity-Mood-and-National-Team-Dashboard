"""把指标、评分、救市资金叠加串成一条流水线；仪表盘和回测共用。"""
from indicators import build_indicators, load_data
from national_team import apply_overlay, build_national_team
from scoring import load_config, score_frame


def run_model(data=None, config=None, fear_weights=None, axis_weights=None, nt_mode=None):
    config = config or load_config()
    data = data if data is not None else load_data()
    ind = build_indicators(data, config)
    scores = score_frame(ind, config, fear_weights, axis_weights)
    nt = build_national_team(ind, data, config)
    comp_adj, mult_adj = apply_overlay(scores["composite"], scores["multiplier"], nt, config, nt_mode)
    frame = ind.join(scores).join(nt.add_prefix("nt_"))
    frame["composite_final"] = comp_adj
    if mult_adj is None:
        from scoring import multiplier_of
        mult_adj = comp_adj.map(lambda s: multiplier_of(s, config))
    frame["multiplier_final"] = mult_adj
    return frame

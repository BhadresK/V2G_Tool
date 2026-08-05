from __future__ import annotations
import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb

# EN 50160 voltage band - the standard low-voltage tolerance most German DSOs work to
V_LIMITS                 = (0.9, 1.1)
TRAFO_RATING_MVA_DEFAULT = 1.0
SIMBENCH_CODE            = "1-LV-semiurb4--0-no_sw"
TRAILER_BUS_NAME = "LV4.101 Bus 44"
MONITORED_BRANCH_NAMES = [
    "LV4.101 Bus 32",
    "LV4.101 Bus 36",
    "LV4.101 Bus 37",
    "LV4.101 Bus 38",
    "LV4.101 Bus 39",
    "LV4.101 Bus 42",
    "LV4.101 Bus 43",
    "LV4.101 Bus 44",
]

_DAY_IDX_DEFAULT = 164

def build_grid(trafo_rating_mva=TRAFO_RATING_MVA_DEFAULT, simbench_code=SIMBENCH_CODE):
    net = sb.get_simbench_net(simbench_code)
    trafo_idx = get_transformer(net)
    net.trafo.at[trafo_idx, "sn_mva"] = float(trafo_rating_mva)
    return net


def get_transformer(net):
    """Index of the (single) MV/LV transformer in this SimBench LV grid."""
    if len(net.trafo) != 1:
        raise RuntimeError(f"Expected exactly 1 transformer, found {len(net.trafo)}.")
    return net.trafo.index[0]


def _name_to_bus(net, name):
    idx = net.bus.index[net.bus["name"] == name].tolist()
    if not idx:
        raise RuntimeError(f"Bus '{name}' not found in network.")
    return idx[0]

def get_trailer_bus(net):
    return _name_to_bus(net, TRAILER_BUS_NAME)

def get_monitored_buses(net):
    return {name: _name_to_bus(net, name) for name in MONITORED_BRANCH_NAMES}


def get_monitored_lines(net):
    buses = get_monitored_buses(net)
    names = MONITORED_BRANCH_NAMES
    segs = []
    for n1, n2 in zip(names[:-1], names[1:]):
        b1, b2 = buses[n1], buses[n2]
        ln = net.line.index[
            ((net.line.from_bus == b1) & (net.line.to_bus == b2))
            | ((net.line.from_bus == b2) & (net.line.to_bus == b1))
        ].tolist()
        if not ln:
            raise RuntimeError(f"No line found between '{n1}' and '{n2}'.")
        segs.append((n1, n2, ln[0]))
    return segs

def simbench_background_profiles(net, simbench_code=SIMBENCH_CODE, day_idx=_DAY_IDX_DEFAULT):
    # SimBench ships full-year profiles at 15-min resolution (96 slots/day)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    p_full = profiles[("load", "p_mw")]
    q_full = profiles[("load", "q_mvar")]

    day_p = p_full.iloc[day_idx * 96:(day_idx + 1) * 96]
    day_q = q_full.iloc[day_idx * 96:(day_idx + 1) * 96]

    hourly_p = day_p.groupby(np.arange(96) // 4).mean()
    hourly_q = day_q.groupby(np.arange(96) // 4).mean()

    # Columns are SimBench's internal (load_name) labels; map back to net.load.index
    hourly_p.columns = net.load.index
    hourly_q.columns = net.load.index
    return hourly_p, hourly_q


def background_total_kw(simbench_code=SIMBENCH_CODE, day_idx=_DAY_IDX_DEFAULT):
    net = sb.get_simbench_net(simbench_code)
    hourly_p, _ = simbench_background_profiles(net, simbench_code, day_idx)
    return (hourly_p.sum(axis=1) * 1000.0).values

def trailer_calendar_profile(result, arrival_h, departure_h, n_slots=24, dt_h=1.0):
    a = int(round(arrival_h   / dt_h)) % n_slots
    d = int(round(departure_h / dt_h)) % n_slots
    window_slots = list(range(a, n_slots)) + list(range(0, d))

    Pc  = np.asarray(result["Pc_w"], dtype=float)
    Pd  = np.asarray(result["Pd_w"], dtype=float)
    tru = result.get("tru_w")
    tru = np.zeros_like(Pc) if tru is None else np.asarray(tru, dtype=float)

    P_net = np.zeros(n_slots)
    for t, idx in enumerate(window_slots):
        if t < len(Pc):
            tru_t = tru[t] if t < len(tru) else 0.0
            P_net[idx] = Pc[t] - Pd[t] + tru_t
    return P_net

def run_grid_timeseries(net, P_trailer_kw, hourly_bg_p_mw, hourly_bg_q_mvar, n_trailers=1):
    if len(P_trailer_kw) != 24:
        raise ValueError("P_trailer_kw must be length 24.")
    if hourly_bg_p_mw.shape[0] != 24 or hourly_bg_q_mvar.shape[0] != 24:
        raise ValueError("Background profiles must have 24 rows.")

    P_trailer_arr = np.asarray(P_trailer_kw, dtype=float)
    if not np.all(np.isfinite(P_trailer_arr)):
        bad_hours = np.where(~np.isfinite(P_trailer_arr))[0].tolist()
        raise ValueError(
            f"P_trailer_kw contains non-finite values at hour(s) {bad_hours}: "
            f"{P_trailer_arr[bad_hours].tolist()}. This means an upstream "
            "MILP/MPC solve returned NaN for at least one trailer before it "
            "was summed into this fleet profile -- check each trailer's "
            "schedule individually rather than the summed total."
        )

    trailer_bus = get_trailer_bus(net)
    trafo_idx = get_transformer(net)
    monitored = get_monitored_buses(net)

    trailer_load_idx = pp.create_load(
        net, bus=trailer_bus, p_mw=0.0, q_mvar=0.0, name="depot_trailer"
    )
    P_trailer_mw = n_trailers * P_trailer_arr / 1000.0

    rows = []
    last_error = None
    for t in range(24):
        net.load.loc[hourly_bg_p_mw.columns, "p_mw"]   = hourly_bg_p_mw.iloc[t].values
        net.load.loc[hourly_bg_q_mvar.columns, "q_mvar"] = hourly_bg_q_mvar.iloc[t].values
        net.load.at[trailer_load_idx, "p_mw"] = float(P_trailer_mw[t])
        net.load.at[trailer_load_idx, "q_mvar"] = 0.0

        try:
            pp.runpp(net, numba=False)
            v_by_bus = {f"v_{name}": float(net.res_bus.at[idx, "vm_pu"])
                        for name, idx in monitored.items()}
            v_min_net = float(net.res_bus["vm_pu"].min())
            v_max_net = float(net.res_bus["vm_pu"].max())
            v_at_trailer = float(net.res_bus.at[trailer_bus, "vm_pu"])

            trafo_load_pct = float(net.res_trafo.at[trafo_idx, "loading_percent"])
            trafo_p_kw     = float(net.res_trafo.at[trafo_idx, "p_hv_mw"]) * 1000.0
            ok = True
        except Exception as e:
            v_by_bus = {f"v_{name}": np.nan for name in monitored}
            v_min_net = v_max_net = v_at_trailer = trafo_load_pct = trafo_p_kw = np.nan
            ok = False
            last_error = f"hour {t}: {type(e).__name__}: {e}"

        P_bg_kw = float(hourly_bg_p_mw.iloc[t].sum() * 1000.0)
        row = {
            "hour":                 t,
            "P_trailer_kw":         float(n_trailers * P_trailer_kw[t]),
            "P_bg_kw":              P_bg_kw,
            "P_total_kw":           P_bg_kw + float(n_trailers * P_trailer_kw[t]),
            "v_min_pu":             v_min_net,
            "v_max_pu":             v_max_net,
            "v_trailer_bus":        v_at_trailer,
            "trafo_load_pct":       trafo_load_pct,
            "trafo_p_kw":           trafo_p_kw,
            "converged":            ok,
        }
        row.update(v_by_bus)
        rows.append(row)

    n_failed = sum(1 for r in rows if not r["converged"])
    if n_failed == 24:
        raise RuntimeError(
            "Power flow failed to converge for ALL 24 hours -- this is a "
            f"systematic problem, not normal non-convergence. Last pandapower "
            f"error: {last_error}"
        )
    elif n_failed > 0:
        import warnings
        warnings.warn(
            f"Power flow failed to converge for {n_failed}/24 hours. "
            f"Last error: {last_error}"
        )

    return pd.DataFrame(rows)

def extract_kpis(ts, v_limits=V_LIMITS, overload_pct=100.0):
    v_lo, v_hi = v_limits
    per_bus_vmin = {
        f"v_min_{name}": float(ts[f"v_{name}"].min())
        for name in MONITORED_BRANCH_NAMES if f"v_{name}" in ts.columns
    }
    kpis = {
        "peak_total_kw":        float(ts["P_total_kw"].max()),
        "min_total_kw":         float(ts["P_total_kw"].min()),
        "energy_imported_kwh":  float(ts["P_total_kw"].clip(lower=0).sum()),
        "energy_exported_kwh":  float(-ts["P_total_kw"].clip(upper=0).sum()),
        "v_min_pu":             float(ts["v_min_pu"].min()),
        "v_max_pu":             float(ts["v_max_pu"].max()),
        "v_violations_low":     int((ts["v_min_pu"] < v_lo).sum()),
        "v_violations_high":    int((ts["v_max_pu"] > v_hi).sum()),
        "trafo_max_loading":    float(ts["trafo_load_pct"].max()),
        "trafo_overload_hours": int((ts["trafo_load_pct"] > overload_pct).sum()),
    }
    kpis.update(per_bus_vmin)
    return kpis


def compare_scenarios(ts_dumb, ts_smart):
    peak_d = float(ts_dumb["P_total_kw"].max())
    peak_s = float(ts_smart["P_total_kw"].max())
    shaved_kw  = peak_d - peak_s
    shaved_pct = 100.0 * shaved_kw / peak_d if peak_d > 1e-9 else 0.0
    return {
        "peak_dumb_kw":    peak_d,
        "peak_smart_kw":   peak_s,
        "peak_shaved_kw":  shaved_kw,
        "peak_shaved_pct": shaved_pct,
    }

if __name__ == "__main__":
    print("grid_sim.py (SimBench) smoke test")
    print("-" * 50)

    net = build_grid(trafo_rating_mva=1.0)
    print("Transformer:", get_transformer(net), "sn_mva=", net.trafo.sn_mva.iloc[0])
    print("Trailer bus:", get_trailer_bus(net), TRAILER_BUS_NAME)
    print("Monitored branch:", get_monitored_buses(net))
    print("Monitored lines:", get_monitored_lines(net))

    hourly_p, hourly_q = simbench_background_profiles(net)
    print("Background total kW by hour:", (hourly_p.sum(axis=1) * 1000).round(1).tolist())

    P_dumb = np.zeros(24)
    P_dumb[16:19] = 22.0 + 6.6
    plug_hours = [16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    for h in plug_hours:
        if P_dumb[h] == 0.0:
            P_dumb[h] = 6.6

    P_smart = np.zeros(24)
    for h in plug_hours:
        P_smart[h] = 6.6
    P_smart[2:5] += 22.0

    print("\n--- Dumb (n_trailers=1) ---")
    ts_d = run_grid_timeseries(build_grid(), P_dumb, hourly_p, hourly_q, n_trailers=1)
    print(extract_kpis(ts_d))

    print("\n--- Smart (n_trailers=1) ---")
    ts_s = run_grid_timeseries(build_grid(), P_smart, hourly_p, hourly_q, n_trailers=1)
    print(extract_kpis(ts_s))

    print("\n--- Comparison ---")
    print(compare_scenarios(ts_d, ts_s))

    print("\n--- Dumb fleet (n_trailers=20) ---")
    ts_d20 = run_grid_timeseries(build_grid(), P_dumb, hourly_p, hourly_q, n_trailers=20)
    print(extract_kpis(ts_d20))
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from v2g import (
    WINTER_M, SUMMER_M,
    V2GParams,
    compose_v2gp_price,
    get_wd_window,
    get_tru_1h_trace,
    run_A_dumb, run_B_smart, run_C_milp, run_D_mpc,
    make_kpi,
    _load_csv_raw,
    _passthrough_profile,
)
from grid_sim import (
    background_total_kw, build_grid, simbench_background_profiles,
    trailer_calendar_profile, run_grid_timeseries, extract_kpis,
    compare_scenarios,
)
from grid_charts import (
    chart_power_stack, chart_voltage, chart_trafo_loading,
    chart_compare, kpi_table_html, COL,
)

CSV_PATH = "2025_Electricity_Price.csv"

FIXED_NET_CT = 6.63 + 1.992 + 0.941 + 0.446 + 2.05 + 1.559 
VAT_RATE = 0.19
FUTURE_A_EXEMPT_CT = 6.63
FUTURE_B_EXEMPT_CT = 6.63 + 2.05 + 0.446 + 0.941 + 1.559


# FIXED 15-TRAILER FLEET TABLE

FLEET_TABLE = [
    {"trailer_id": 1,  "arrival_h": 14.0, "departure_h": 5.0,  "soc_init": 85, "soc_dep": 100},
    {"trailer_id": 2,  "arrival_h": 14.0, "departure_h": 5.0,  "soc_init": 65, "soc_dep": 100},
    {"trailer_id": 3,  "arrival_h": 14.0, "departure_h": 5.0,  "soc_init": 50, "soc_dep": 100},
    {"trailer_id": 4,  "arrival_h": 14.0, "departure_h": 5.0,  "soc_init": 40, "soc_dep": 100},
    {"trailer_id": 5,  "arrival_h": 14.0, "departure_h": 5.0,  "soc_init": 20, "soc_dep": 100},
    {"trailer_id": 6,  "arrival_h": 18.0, "departure_h": 8.0,  "soc_init": 85, "soc_dep": 100},
    {"trailer_id": 7,  "arrival_h": 18.0, "departure_h": 8.0,  "soc_init": 65, "soc_dep": 100},
    {"trailer_id": 8,  "arrival_h": 18.0, "departure_h": 8.0,  "soc_init": 50, "soc_dep": 100},
    {"trailer_id": 9,  "arrival_h": 18.0, "departure_h": 8.0,  "soc_init": 40, "soc_dep": 100},
    {"trailer_id": 10, "arrival_h": 18.0, "departure_h": 8.0,  "soc_init": 20, "soc_dep": 100},
    {"trailer_id": 11, "arrival_h": 21.0, "departure_h": 11.0, "soc_init": 85, "soc_dep": 100},
    {"trailer_id": 12, "arrival_h": 21.0, "departure_h": 11.0, "soc_init": 65, "soc_dep": 100},
    {"trailer_id": 13, "arrival_h": 21.0, "departure_h": 11.0, "soc_init": 50, "soc_dep": 100},
    {"trailer_id": 14, "arrival_h": 21.0, "departure_h": 11.0, "soc_init": 40, "soc_dep": 100},
    {"trailer_id": 15, "arrival_h": 21.0, "departure_h": 11.0, "soc_init": 20, "soc_dep": 100},
]

URGENCY_ORDER = sorted(
    range(len(FLEET_TABLE)),
    key=lambda i: (FLEET_TABLE[i]["soc_init"], FLEET_TABLE[i]["trailer_id"]),
)

@st.cache_data(show_spinner=False)
def _fleet_seasonal_profile(months: tuple, is_weekend: bool) -> np.ndarray:
    df = _load_csv_raw(CSV_PATH)
    mask = df["month"].isin(list(months)) & (df["is_weekend"] == is_weekend)
    sub = df[mask]
    if len(sub) == 0:
        raise ValueError(f"No data for months={months}, weekend={is_weekend}")
    profile = sub.groupby("slot")["price"].mean().values
    return _passthrough_profile(profile)


@st.cache_data(show_spinner=False)
def run_fleet_for_season(season: str, tru_cycle: str, aux_power_w: int,
                          bat_heat_w: int, peak_shaving: bool,
                          lambda_ps: float = 2.0):

    months = WINTER_M if season == "winter" else SUMMER_M
    buy_full  = _fleet_seasonal_profile(tuple(months), False)
    v2gp_full = compose_v2gp_price(buy_full, exempt_ct=0.0)

    is_winter    = (season == "winter")
    aux_kw       = aux_power_w / 1000.0
    heat_kw      = (bat_heat_w / 1000.0) if is_winter else 0.0
    parasitic_kw = aux_kw + heat_kw

    v2g = V2GParams(soc_departure_pct=100.0)   
    P_bg_site = background_total_kw()

    fleet_profiles = {"A": np.zeros(24), "B": np.zeros(24),
                      "C": np.zeros(24), "D": np.zeros(24)}
    committed = {"B": P_bg_site.copy(), "C": P_bg_site.copy()}
    per_trailer_kpi = {"A": [], "B": [], "C": [], "D": []}
    trailer_c_profiles = [None] * len(FLEET_TABLE)

    # A
    for t in FLEET_TABLE:
        E_init = v2g.usable_capacity_kWh * t["soc_init"] / 100.0
        win, arr, dep, W = get_wd_window(v2g, t["arrival_h"], t["departure_h"])
        buy_w, v2gp_w = buy_full[win], v2gp_full[win]
        tru_w = get_tru_1h_trace(tru_cycle, W, v2g.dt_h) + parasitic_kw

        Pc, Pd, soc = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, tru_w)
        r = make_kpi(f"A (T{t['trailer_id']})", v2g, Pc, Pd, soc,
                     buy_w, v2gp_w, E_init, arr_disp=0, dep_disp=W,
                     tru_w=tru_w, P_bg_full=P_bg_site)
        per_trailer_kpi["A"].append({"trailer_id": t["trailer_id"], **r})
        prof = trailer_calendar_profile(r, t["arrival_h"], t["departure_h"])
        fleet_profiles["A"] = fleet_profiles["A"] + prof

    # B, C
    for scenario_key, runner in (("B", run_B_smart), ("C", run_C_milp)):
        for i in URGENCY_ORDER:
            t = FLEET_TABLE[i]
            E_init = v2g.usable_capacity_kWh * t["soc_init"] / 100.0
            win, arr, dep, W = get_wd_window(v2g, t["arrival_h"], t["departure_h"])
            buy_w, v2gp_w = buy_full[win], v2gp_full[win]
            tru_w = get_tru_1h_trace(tru_cycle, W, v2g.dt_h) + parasitic_kw

            P_bg_full_ref = committed[scenario_key]
            P_bg_w_ref    = P_bg_full_ref[np.array(win) % 24]

            Pc, Pd, soc = runner(v2g, buy_w, v2gp_w, E_init, tru_w,
                                 peak_shaving=peak_shaving,
                                 P_bg_w=P_bg_w_ref, P_bg_full=P_bg_full_ref,
                                 lambda_ps=lambda_ps)
            r = make_kpi(f"{scenario_key} (T{t['trailer_id']})", v2g, Pc, Pd, soc,
                         buy_w, v2gp_w, E_init, arr_disp=0, dep_disp=W,
                         tru_w=tru_w, P_bg_full=P_bg_full_ref)
            per_trailer_kpi[scenario_key].append({"trailer_id": t["trailer_id"], **r})

            prof = trailer_calendar_profile(r, t["arrival_h"], t["departure_h"])
            fleet_profiles[scenario_key] = fleet_profiles[scenario_key] + prof
            committed[scenario_key] = committed[scenario_key] + prof

            if scenario_key == "C":
                trailer_c_profiles[i] = prof

    # D
    fleet_c_total = fleet_profiles["C"]
    for i, t in enumerate(FLEET_TABLE):
        E_init = v2g.usable_capacity_kWh * t["soc_init"] / 100.0
        win, arr, dep, W = get_wd_window(v2g, t["arrival_h"], t["departure_h"])
        buy_w, v2gp_w = buy_full[win], v2gp_full[win]
        tru_w = get_tru_1h_trace(tru_cycle, W, v2g.dt_h) + parasitic_kw

        others_c = fleet_c_total - trailer_c_profiles[i]
        P_bg_full_ref = P_bg_site + others_c
        P_bg_w_ref    = P_bg_full_ref[np.array(win) % 24]

        Pc, Pd, soc = run_D_mpc(v2g, buy_w, v2gp_w, E_init, tru_w,
                                peak_shaving=peak_shaving,
                                P_bg_w=P_bg_w_ref, P_bg_full=P_bg_full_ref,
                                lambda_ps=lambda_ps)
        r = make_kpi(f"D (T{t['trailer_id']})", v2g, Pc, Pd, soc,
                     buy_w, v2gp_w, E_init, arr_disp=0, dep_disp=W,
                     tru_w=tru_w, P_bg_full=P_bg_full_ref)
        per_trailer_kpi["D"].append({"trailer_id": t["trailer_id"], **r})
        prof = trailer_calendar_profile(r, t["arrival_h"], t["departure_h"])
        fleet_profiles["D"] = fleet_profiles["D"] + prof

    return {"fleet_profiles": fleet_profiles, "per_trailer_kpi": per_trailer_kpi}


# FLEET ECONOMIC KPI TABLE

def fleet_kpi_table_html(per_trailer_kpi, fixed_price_eur_kwh, peak_shaved_kw=None,
                         title="Fleet Economic KPI"):
    peak_shaved_kw = peak_shaved_kw or {}
    LEISTUNGSPREIS_EUR_KW_YEAR = 96.52

    def _charge_allin(r, exempt_ct):
        if r["charge_kwh"] < 1e-9:
            return 0.0
        exempt_kwh     = min(r["charge_kwh"], r["v2g_kwh"])
        non_exempt_kwh = r["charge_kwh"] - exempt_kwh
        vat_gross      = 1.0 + VAT_RATE
        return (r["charge_cost"] * vat_gross
                + (non_exempt_kwh * FIXED_NET_CT
                   + exempt_kwh * (FIXED_NET_CT - exempt_ct)) / 100.0 * vat_gross)

    def _row_for_trailer(r, exempt_ct, is_fixed=False):
        if is_fixed:
            charge   = r["charge_kwh"] * fixed_price_eur_kwh
            v2g_rev  = 0.0
            deg_cost = r.get("deg_cost_eur", 0.0)
        else:
            charge   = _charge_allin(r, exempt_ct)
            avg_spot = (r["charge_cost"] / r["charge_kwh"]
                       if r["charge_kwh"] > 1e-9 else 0.10)
            v2g_rev  = max(0.0, r.get("v2g_rev", r["v2g_kwh"] * avg_spot))
            deg_cost = r.get("deg_cost_eur", 0.0)
        net_no_deg = charge - v2g_rev
        return charge, v2g_rev, deg_cost, net_no_deg, net_no_deg + deg_cost

    scenario_order  = ["F", "A", "B", "C", "D"]
    scenario_labels = {"F": "F - Fixed Price", "A": "A - Dumb", "B": "B - Smart",
                       "C": "C - MILP", "D": "D - MPC"}
    reg_scenarios = [("cur", 0.0), ("fa", FUTURE_A_EXEMPT_CT), ("fb", FUTURE_B_EXEMPT_CT)]

    rows_html = ""
    for sc in scenario_order:
        is_fixed = (sc == "F")
        trailer_rows = per_trailer_kpi.get("A" if is_fixed else sc, [])
        if not trailer_rows:
            continue

        totals = {}
        for key, exempt_ct in reg_scenarios:
            c_sum = v_sum = d_sum = 0.0
            for r in trailer_rows:
                c, v, d, _, _ = _row_for_trailer(r, exempt_ct, is_fixed)
                c_sum += c; v_sum += v; d_sum += d
            credit = peak_shaved_kw.get(sc, 0.0) * LEISTUNGSPREIS_EUR_KW_YEAR / 365.0
            n_sum = c_sum - v_sum - credit
            totals[key] = (c_sum, v_sum, d_sum, n_sum, n_sum + d_sum)

        cur, fa, fb = totals["cur"], totals["fa"], totals["fb"]
        rows_html += (
            f"<tr><td style='text-align:left;font-weight:600'>{scenario_labels[sc]}</td>"
            f"<td>{cur[0]:.1f}</td><td>{cur[1]:.1f}</td><td>{cur[2]:.1f}</td><td>{cur[3]:.1f}</td><td style='font-weight:600'>{cur[4]:.1f}</td>"
            f"<td>{fa[0]:.1f}</td><td>{fa[1]:.1f}</td><td>{fa[2]:.1f}</td><td>{fa[3]:.1f}</td><td style='font-weight:600'>{fa[4]:.1f}</td>"
            f"<td>{fb[0]:.1f}</td><td>{fb[1]:.1f}</td><td>{fb[2]:.1f}</td><td>{fb[3]:.1f}</td><td style='font-weight:600'>{fb[4]:.1f}</td>"
            f"</tr>"
        )

    ch, vr, dc, nc, nwd = "Charge (\u20ac/d)", "V2G Rev (\u20ac/d)", "Deg Cost (\u20ac/d)", "Net (\u20ac/d)", "Net+Deg (\u20ac/d)"
    return f"""
<style>
.fleet_kpi {{ border-collapse:collapse; width:100%; font-size:11px; margin-top:6px; }}
.fleet_kpi th, .fleet_kpi td {{ border:1px solid #d0d0d0; padding:4px 8px; text-align:center; white-space:nowrap; }}
.fleet_kpi th {{ background:#f0f2f6; font-weight:600; }}
.fleet_kpi tbody tr:hover td {{ background:#eef4ff; }}
.fleet_kpi th.cur {{ background:#e8eaf6; }}
.fleet_kpi th.fa  {{ background:#e3f2fd; }}
.fleet_kpi th.fb  {{ background:#f3e5f5; }}
</style>
<b>{title}</b>
<table class='fleet_kpi'>
<thead>
  <tr>
    <th rowspan='2'>Scenario</th>
    <th colspan='5' class='cur'>Current</th>
    <th colspan='5' class='fa'>Future A</th>
    <th colspan='5' class='fb'>Future B</th>
  </tr>
  <tr>
    <th class='cur'>{ch}</th><th class='cur'>{vr}</th><th class='cur'>{dc}</th><th class='cur'>{nc}</th><th class='cur'>{nwd}</th>
    <th class='fa'>{ch}</th><th class='fa'>{vr}</th><th class='fa'>{dc}</th><th class='fa'>{nc}</th><th class='fa'>{nwd}</th>
    <th class='fb'>{ch}</th><th class='fb'>{vr}</th><th class='fb'>{dc}</th><th class='fb'>{nc}</th><th class='fb'>{nwd}</th>
  </tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
"""

def _render_fleet_season_block(season_label, season_key, trafo_mva, tru_cycle,
                                 aux_power_w, bat_heat_w, peak_shaving, fixed_price,
                                 lambda_ps=2.0):
    with st.spinner(f"Running 15-trailer fleet -- {season_label} (A/B/C/D, coordinated)..."):
        fleet = run_fleet_for_season(season_key, tru_cycle, aux_power_w,
                                      bat_heat_w, peak_shaving, lambda_ps=lambda_ps)

    net = build_grid(trafo_rating_mva=trafo_mva)
    hourly_p, hourly_q = simbench_background_profiles(net)

    scenario_labels = {"A": "A - Dumb", "B": "B - Smart", "C": "C - MILP", "D": "D - MPC"}
    ts_map = {}
    for sc in ("A", "B", "C", "D"):
        ts_map[sc] = run_grid_timeseries(
            build_grid(trafo_rating_mva=trafo_mva),
            fleet["fleet_profiles"][sc], hourly_p, hourly_q, n_trailers=1,
        )

    st.markdown(f"### {season_label}")

    for sc in ("A", "B", "C", "D"):
        st.subheader(scenario_labels[sc])
        ts = ts_map[sc]
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"{scenario_labels[sc]} : Fleet Depot Power (15 trailers)")
            st.altair_chart(chart_power_stack(ts), use_container_width=True)
            st.caption(f"{scenario_labels[sc]} : Transformer Loading")
            st.altair_chart(chart_trafo_loading(ts), use_container_width=True)
        with col2:
            st.caption(f"{scenario_labels[sc]} : Bus 44 Voltage")
            st.altair_chart(chart_voltage(ts), use_container_width=True)
        st.markdown("")

    st.markdown(f"**Peak Shaving vs Scenario A -- {season_label}**")
    _CMP_COLOR = {"B": COL["smart"], "C": COL["milp"], "D": COL["mpc"]}
    _peak_shaved_kw = {}
    for sc in ("C", "D"):
        comp = compare_scenarios(ts_map["A"], ts_map[sc])
        _peak_shaved_kw[sc] = comp["peak_shaved_kw"]
        st.caption(f"A - Dumb vs {scenario_labels[sc]}")
        st.altair_chart(
            chart_compare(ts_map["A"], ts_map[sc],
                          label_smart=scenario_labels[sc],
                          color_smart=_CMP_COLOR[sc]),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Peak change vs A", f"{comp['peak_shaved_kw']:.1f} kW",
                  f"{comp['peak_shaved_pct']:.1f} %")
        c2.metric("Peak (Dumb)", f"{comp['peak_dumb_kw']:.1f} kW")

    st.markdown(f"**Grid KPI Summary -- {season_label}**")
    grid_kpis = {scenario_labels[sc]: extract_kpis(ts_map[sc]) for sc in ("A", "B", "C", "D")}
    st.markdown(kpi_table_html(grid_kpis, title=f"Grid impact KPIs -- {season_label}"),
                unsafe_allow_html=True)

    st.markdown(f"**Economic KPI Summary -- {season_label}**")
    st.markdown(
        fleet_kpi_table_html(fleet["per_trailer_kpi"], fixed_price,
                             peak_shaved_kw=_peak_shaved_kw,
                             title=f"Fleet Economic KPI -- {season_label}"),
        unsafe_allow_html=True,
    )
    st.markdown("---")

def render_fleet_grid_page(trafo_mva, tru_cycle, aux_power_w, bat_heat_w,
                            peak_shaving, fixed_price, lambda_ps=2.0):
    st.title("LV Grid Impact -- Fixed 15-Trailer Fleet")
    
    _render_fleet_season_block("Winter", "winter", trafo_mva, tru_cycle,
                                aux_power_w, bat_heat_w, peak_shaving, fixed_price,
                                lambda_ps=lambda_ps)
    _render_fleet_season_block("Summer", "summer", trafo_mva, tru_cycle,
                                aux_power_w, bat_heat_w, peak_shaving, fixed_price,
                                lambda_ps=lambda_ps)
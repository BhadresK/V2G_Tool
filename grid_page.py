from __future__ import annotations
import streamlit as st
from pathlib import Path
import numpy as np
from network_diagram_patch import render_static_network_diagram

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
    build_grid, background_total_kw, simbench_background_profiles,
    trailer_calendar_profile,
    run_grid_timeseries, extract_kpis, compare_scenarios,
    TRAFO_RATING_MVA_DEFAULT,
)

from grid_charts import (
    chart_power_stack, chart_voltage, chart_trafo_loading,
    chart_scenario_comparison,
    chart_compare, kpi_table_html, COL,
)
from fleet_sim import render_fleet_grid_page

CSV_PATH = "2025_Electricity_Price.csv"

@st.cache_data(show_spinner=False)
def _seasonal_profile(months, is_weekend):
    df   = _load_csv_raw(CSV_PATH)
    mask = df["month"].isin(list(months)) & (df["is_weekend"] == is_weekend)
    sub  = df[mask]
    if len(sub) == 0:
        raise ValueError(f"No data for months={months}, weekend={is_weekend}")
    profile = sub.groupby("slot")["price"].mean().values
    return _passthrough_profile(profile)

def _run_one(scenario_key, v2g, buy_w, v2gp_w, E_init, tru_w, W,
             peak_shaving=False, P_bg_w=None, P_bg_full=None):
    """Run a single scenario (A/B/C/D) and return its make_kpi() dict."""
    ps_kwargs = dict(peak_shaving=peak_shaving, P_bg_w=P_bg_w, P_bg_full=P_bg_full)
    if scenario_key == "A":
        Pc, Pd, soc = run_A_dumb(v2g, buy_w, v2gp_w, W, E_init, tru_w)
        label = "A - Dumb"
    elif scenario_key == "B":
        Pc, Pd, soc = run_B_smart(v2g, buy_w, v2gp_w, E_init, tru_w, **ps_kwargs)
        label = "B - Smart"
    elif scenario_key == "C":
        Pc, Pd, soc = run_C_milp(v2g, buy_w, v2gp_w, E_init, tru_w, **ps_kwargs)
        label = "C - MILP"
    elif scenario_key == "D":
        Pc, Pd, soc = run_D_mpc(v2g, buy_w, v2gp_w, E_init, tru_w, **ps_kwargs)
        label = "D - MPC"
    else:
        raise ValueError(f"Unknown scenario key: {scenario_key}")

    return make_kpi(
        label, v2g, Pc, Pd, soc, buy_w, v2gp_w, E_init,
        arr_disp=0, dep_disp=W, tru_w=tru_w, P_bg_full=P_bg_full,
    )

@st.cache_data(show_spinner=False)
def run_grid_for_scenario(
    season, scenario_key,
    arrival_h, departure_h,
    soc_init, soc_dep, tru_cycle,
    aux_power_w, bat_heat_w,
    n_trailers, trafo_mva,
    peak_shaving=False,
):
    """Compose: run v2g scenario -> build calendar profile -> grid time-series."""
    months = WINTER_M if season == "winter" else SUMMER_M
    buy  = _seasonal_profile(tuple(months), is_weekend=False)
    v2gp = compose_v2gp_price(buy, exempt_ct=0.0)
    v2g  = V2GParams(soc_departure_pct=float(soc_dep))
    E_init = v2g.usable_capacity_kWh * soc_init / 100.0

    win, _arr, _dep, W = get_wd_window(v2g, arrival_h, departure_h)
    buy_w  = buy[win]
    v2gp_w = v2gp[win]

    is_winter   = (season == "winter")
    aux_kw      = aux_power_w / 1000.0
    heat_kw     = (bat_heat_w / 1000.0) if is_winter else 0.0
    parasitic_kw = aux_kw + heat_kw
    tru_w = get_tru_1h_trace(tru_cycle, W, v2g.dt_h) + parasitic_kw
    P_bg_full = background_total_kw()
    P_bg_w = P_bg_full[np.array(win) % 24]

    result = _run_one(scenario_key, v2g, buy_w, v2gp_w, E_init, tru_w, W,
                      peak_shaving=peak_shaving, P_bg_w=P_bg_w, P_bg_full=P_bg_full)

    P_trailer = trailer_calendar_profile(result, arrival_h, departure_h)

    net = build_grid(trafo_rating_mva=trafo_mva)
    hourly_p, hourly_q = simbench_background_profiles(net)
    ts = run_grid_timeseries(net, P_trailer, hourly_p, hourly_q, n_trailers=n_trailers)

    return result, ts

def render_grid_page():
    st.title("LV Grid Impact")

    if not Path(CSV_PATH).exists():
        st.error(f"Price CSV '{CSV_PATH}' not found in working directory.")
        st.stop()

    with st.sidebar:
        st.header("Grid Simulation Inputs")

        st.subheader("Simulation Mode")
        sim_mode = st.radio(
            "Mode",
            ["Single representative trailer", "Fixed 15-trailer fleet"],
            index=0,
        )

        st.subheader("Grid")
        trafo_mva = st.number_input(
            "Depot transformer rating (MVA)",
            min_value=0.1, max_value=5.0,
            value=float(TRAFO_RATING_MVA_DEFAULT), step=0.1,
        )

        if sim_mode == "Single representative trailer":
            st.subheader("Season")
            season = st.radio("Season", ["winter", "summer"], index=0)

            st.subheader("Fleet (Stage 2)")
            n_trailers = st.slider("Number of trailers at depot", 1, 50, 1)

            st.subheader("Scenarios to Run")
            do_A = st.checkbox("A -- Dumb",  True)
            do_B = st.checkbox("B -- Smart", False)
            do_C = st.checkbox("C -- MILP",  True)
            do_D = st.checkbox("D -- MPC",   False)
        else:
            st.subheader("Fleet Peak-Shaving Weight")
            _cfg_ps = st.session_state.get("cfg", {})
            _cfg_ps["lambda_ps_fleet"] = st.slider(
                "Fleet lambda_ps (EUR/kW/day)",
                min_value=0.2, max_value=10.0,
                value=float(_cfg_ps.get("lambda_ps_fleet", 2.0)), step=0.2,
            )
            st.session_state.cfg = _cfg_ps

    if sim_mode == "Fixed 15-trailer fleet":
        _cfg = st.session_state.get("cfg", {})
        render_fleet_grid_page(
            trafo_mva=trafo_mva,
            tru_cycle=str(_cfg.get("tru_cycle", "OFF")),
            aux_power_w=int(_cfg.get("aux_power_w", 400)),
            bat_heat_w=int(_cfg.get("bat_heat_w", 100)),
            peak_shaving=True,
            fixed_price=float(_cfg.get("fixed_price", 0.35)),
            lambda_ps=float(_cfg.get("lambda_ps_fleet", 2.0)),
        )
        st.stop()

    _cfg        = st.session_state.get("cfg", {})
    arrival_h   = float(_cfg.get("arrival_str", "16:00").split(":")[0])
    departure_h = float(_cfg.get("departure_str", "06:00").split(":")[0])
    soc_init    = float(_cfg.get("soc_winter", 80) if season == "winter"
                        else _cfg.get("soc_summer", 40))
    soc_dep     = float(_cfg.get("soc_departure", 100))
    tru_cycle   = str(_cfg.get("tru_cycle", "OFF"))
    aux_power_w = int(_cfg.get("aux_power_w", 400))
    bat_heat_w  = int(_cfg.get("bat_heat_w", 100))

    keys_to_run = [k for k, on in [("A", do_A), ("B", do_B),
                                    ("C", do_C), ("D", do_D)] if on]
    if not keys_to_run:
        st.warning("Enable at least one scenario in the sidebar.")
        st.stop()

    # RUN ALL SELECTED SCENARIOS
    results = {}
    ts_map  = {}
    with st.spinner(f"Running {len(keys_to_run)} scenarios..."):
        for k in keys_to_run:
            try:
                _ps_grid = bool(st.session_state.get("cfg", {}).get("peak_shaving", False))
                r, ts = run_grid_for_scenario(
                    season, k, float(arrival_h), float(departure_h),
                    float(soc_init), float(soc_dep), tru_cycle,
                    int(aux_power_w), int(bat_heat_w),
                    int(n_trailers), float(trafo_mva),
                    peak_shaving=_ps_grid,
                )
                results[k] = r
                ts_map[k]  = ts
            except Exception as e:
                st.error(f"Scenario {k} failed: {e}")

    if not ts_map:
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Fleet size",  f"{n_trailers} trailer(s)")
    m2.metric("Trafo",       f"{trafo_mva:.2f} MVA")
    m3.metric("Season",      season.capitalize())
    st.markdown("---")

    with st.expander("Network One-Line Diagram — SimBench 1-LV-semiurb4", expanded=False):
        render_static_network_diagram(trafo_mva)
            
    scenario_labels = {"A": "A - Dumb", "B": "B - Smart",
                        "C": "C - MILP", "D": "D - MPC"}

    for k in keys_to_run:
        st.subheader(scenario_labels[k])
        ts = ts_map[k]
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"{scenario_labels[k]} : Depot Power")
            st.altair_chart(chart_power_stack(ts), use_container_width=True)
            st.caption(f"{scenario_labels[k]} : Transformer Loading")
            st.altair_chart(chart_trafo_loading(ts), use_container_width=True)
        with col2:
            st.caption(f"{scenario_labels[k]} : Bus Voltages")
            st.altair_chart(chart_voltage(ts), use_container_width=True)
        st.markdown("")

    st.markdown("---")

    kpis = {scenario_labels[k]: extract_kpis(ts_map[k]) for k in keys_to_run}
    comp_for_table = None
    if "A" in ts_map and "C" in ts_map:
        comp_for_table = compare_scenarios(ts_map["A"], ts_map["C"])

    ps_col, sc_col = st.columns(2)

    with ps_col:

        if "A" in ts_map and any(k in ts_map for k in ("B", "C", "D")):
            st.subheader("Peak Shaving & Flexibility vs Scenario A")
            _CMP_COLOR = {"B": COL["smart"], "C": COL["milp"], "D": COL["mpc"]}
            for k in ("B", "C", "D"):
                if k not in ts_map:
                    continue
                st.altair_chart(
                    chart_compare(
                        ts_map["A"], ts_map[k],
                        label_smart=scenario_labels[k],
                        color_smart=_CMP_COLOR[k],
                        title=f"A - Dumb vs {scenario_labels[k]} : Total Depot Power",
                    ),
                    use_container_width=True,
                )
                comp = compare_scenarios(ts_map["A"], ts_map[k])
                c1, c2, c3 = st.columns(3)
                c1.metric("Peak shaved",
                        f"{comp['peak_shaved_kw']:.1f} kW",
                        f"{comp['peak_shaved_pct']:.1f} %")
                c2.metric("Peak (Dumb)",  f"{comp['peak_dumb_kw']:.1f} kW")
                c3.metric("Flexibility",  f"{results[k]['v2g_kwh']:.1f} kWh")
                st.markdown("")

    with sc_col:
        st.subheader("Scenario Comparison — Key Metrics")
        st.altair_chart(
            chart_scenario_comparison(kpis),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Grid KPI Summary")
    st.markdown(
        kpi_table_html(kpis, comp_for_table, title="Grid impact KPIs"),
        unsafe_allow_html=True,
    )
if __name__ == "__main__":
    st.set_page_config(page_title="V2G Grid Impact", layout="wide")
    render_grid_page()
from __future__ import annotations
import numpy as np
import pandas as pd
import altair as alt
from grid_sim import MONITORED_BRANCH_NAMES

COL = {
    "trailer":  "#FF7700",
    "bg":       "#999999",
    "total":    "#1565C0",
    "dumb":     "#999999",
    "smart":    "#00ACC1",
    "milp":     "#00BCD4",
    "mpc":      "#FF7700",
    "v_low":    "#C62828",
    "v_high":   "#C62828",
    "v_bus":    "#1B5E20",
    "v_band":   "#E8F5E9",
    "trafo":    "#6A1B9A",
    "overload": "#C62828",
}

V_LIMITS_DEFAULT = (0.9, 1.1)


def _hour_axis():
    return alt.X(
        "hour:Q",
        scale=alt.Scale(domain=[0, 24]),
        axis=alt.Axis(values=list(range(0, 25, 2)),
                      labelExpr="(datum.value<10?'0':'')+datum.value+':00'",
                      labelAngle=-35, title="Hour of day"),
    )


def chart_power_stack(ts, title="Depot power vs time"):
    """Stacked bars (background + trailer) with a labelled total power line."""
    label_map = {"P_trailer_kw": "Trailer (net)", "P_bg_kw": "SimBench LV grid (41 real loads)"}
    stack_order = [label_map["P_bg_kw"], label_map["P_trailer_kw"]]
    total_label = "Total depot power"
    full_domain = stack_order + [total_label]
    full_range  = [COL["bg"], COL["trailer"], COL["total"]]
    color_scale = alt.Scale(domain=full_domain, range=full_range)

    df = ts[["hour", "P_trailer_kw", "P_bg_kw"]].melt(
        id_vars="hour", var_name="series", value_name="kW"
    )
    df["series"] = df["series"].map(label_map)

    pos_stack = ts[["P_trailer_kw", "P_bg_kw"]].clip(lower=0).sum(axis=1)
    neg_stack = ts[["P_trailer_kw", "P_bg_kw"]].clip(upper=0).sum(axis=1)
    y_min = float(min(neg_stack.min(), ts["P_total_kw"].min(), 0.0))
    y_max = float(max(pos_stack.max(), ts["P_total_kw"].max(), 0.0))
    padding = max((y_max - y_min) * 0.15, 2.0)
    y_lo, y_hi = y_min - padding, y_max + padding

    bars = (
        alt.Chart(df).mark_bar(opacity=0.85, size=18).encode(
            x=_hour_axis(),
            y=alt.Y("kW:Q", stack="zero",
                    axis=alt.Axis(title="Power (kW)"),
                    scale=alt.Scale(domain=[y_lo, y_hi])),
            color=alt.Color("series:N", scale=color_scale,
                             legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["hour:Q", "series:N",
                     alt.Tooltip("kW:Q", format=".1f")],
        )
    )

    total_df = ts[["hour", "P_total_kw"]].copy()
    total_df["series"] = total_label

    total_line = (
        alt.Chart(total_df).mark_line(
            interpolate="step-after", strokeWidth=2.0
        ).encode(
            x=_hour_axis(),
            y=alt.Y("P_total_kw:Q", scale=alt.Scale(domain=[y_lo, y_hi])),
            color=alt.Color("series:N", scale=color_scale,
                             legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("P_total_kw:Q", format=".1f", title="Total kW"),
                     alt.Tooltip("hour:Q", title="Hour")],
        )
    )

    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#666", strokeDash=[3, 2], opacity=0.6
    ).encode(y="y:Q")

    return (bars + total_line + zero).properties(
        height=280, background="#FAFAFA"
    ).interactive().configure(padding={"top": 8})

def chart_voltage(ts, v_limits=V_LIMITS_DEFAULT, title="Bus Voltages"):
    """Bus 44 (depot connection / worst-case node) voltage only -- no legend."""
    v_lo, v_hi = v_limits

    band = alt.Chart(pd.DataFrame({
        "y_low":  [v_lo], "y_high": [v_hi],
    })).mark_rect(color=COL["v_band"], opacity=0.40).encode(
        y=alt.Y("y_low:Q",
                scale=alt.Scale(domain=[min(0.85, v_lo - 0.02),
                                        max(1.10, v_hi + 0.02)]),
                axis=alt.Axis(title="Voltage (p.u.)")),
        y2="y_high:Q",
    )

    bus44_col = f"v_{MONITORED_BRANCH_NAMES[-1]}"   # "v_LV4.101 Bus 44"
    if bus44_col not in ts.columns:
        raise KeyError(
            f"chart_voltage: '{bus44_col}' not found in ts. "
            f"Available columns: {list(ts.columns)}"
        )

    plot_col = "bus44_vpu"
    ts_plot = ts[["hour", bus44_col]].rename(columns={bus44_col: plot_col})

    line = alt.Chart(ts_plot).mark_line(
        interpolate="step-after", color=COL["v_bus"], strokeWidth=2.2
    ).encode(
        x=_hour_axis(),
        y=alt.Y(f"{plot_col}:Q",
                axis=alt.Axis(title="Voltage (p.u.)"),
                scale=alt.Scale(domain=[min(0.85, v_lo - 0.02),
                                        max(1.10, v_hi + 0.02)])),
        tooltip=[alt.Tooltip(f"{plot_col}:Q", format=".4f", title="Bus 44 (p.u.)"),
                 "hour:Q"],
    )

    rule_lo = alt.Chart(pd.DataFrame({"y": [v_lo]})).mark_rule(
        color=COL["v_low"], strokeDash=[2, 2], opacity=0.7
    ).encode(y="y:Q")
    rule_hi = alt.Chart(pd.DataFrame({"y": [v_hi]})).mark_rule(
        color=COL["v_high"], strokeDash=[2, 2], opacity=0.7
    ).encode(y="y:Q")

    return (band + line + rule_lo + rule_hi).properties(
        height=260, background="#FAFAFA"
    ).interactive().configure(padding={"top": 8})

def chart_trafo_loading(ts, overload_pct=100.0, title="Depot Transformer Loading"):
    """Loading % over time, with 100% overload reference and area fill."""
    base = alt.Chart(ts).encode(x=_hour_axis())

    # Dynamic y-max: at least 10%, or 120% of actual max, whichever is larger
    y_max = max(10.0, float(ts["trafo_load_pct"].max()) * 1.2)

    area = base.mark_area(
        interpolate="step-after", color=COL["trafo"], opacity=0.30
    ).encode(
        y=alt.Y("trafo_load_pct:Q",
                axis=alt.Axis(title="Loading (% of rated MVA)"),
                scale=alt.Scale(zero=True, domain=[0, y_max])),
    )
    line = base.mark_line(
        interpolate="step-after", color=COL["trafo"], strokeWidth=2.2
    ).encode(
        y=alt.Y("trafo_load_pct:Q",
                scale=alt.Scale(zero=True, domain=[0, y_max])),
        tooltip=[alt.Tooltip("trafo_load_pct:Q", format=".2f", title="Loading %"),
                 alt.Tooltip("trafo_p_kw:Q", format=".1f", title="Trafo P (kW)"),
                 alt.Tooltip("hour:Q", title="Hour")],
    )

    rule_data = pd.DataFrame({"y": [min(overload_pct, y_max * 0.98)]})
    rule = alt.Chart(rule_data).mark_rule(
        color=COL["overload"], strokeDash=[4, 2], strokeWidth=1.5
    ).encode(y="y:Q")

    return (area + line + rule).properties(
        height=240, background="#FAFAFA"
    ).interactive().configure(padding={"top": 8}).configure_view(continuousWidth=300)

def chart_scenario_comparison(kpis_dict, title="Scenario Comparison — Key Grid KPIs"):
    """Compact grouped bar chart comparing key KPIs across scenarios."""
    metrics = [
        ("peak_total_kw",     "Peak (kW)"),
        ("trafo_max_loading", "Trafo (%)"),
        ("v_min_pu",          "V min (p.u.)"),
    ]

    rows = []
    for sc_label, kpi in kpis_dict.items():
        for key, metric_label in metrics:
            rows.append({
                "Scenario": sc_label,
                "Metric":   metric_label,
                "Value":    float(kpi[key]),
            })

    df = pd.DataFrame(rows)

    sc_labels = list(kpis_dict.keys())
    palette   = ["#999999", "#00ACC1", "#FF7700", "#6A1B9A"]
    sc_colors = palette[: len(sc_labels)]

    bars = (
        alt.Chart(df)
        .mark_bar(opacity=0.88, size=10)
        .encode(
            x=alt.X(
                "Metric:N",
                sort=[m[1] for m in metrics],
                axis=alt.Axis(
                    title=None,
                    labelAngle=-35,
                    labelFontSize=9,
                    labelLimit=55,
                ),
            ),
            xOffset=alt.XOffset(
                "Scenario:N",
                sort=sc_labels,
            ),
            y=alt.Y(
                "Value:Q",
                axis=alt.Axis(
                    title=None,
                    labelFontSize=9,
                    titleFontSize=9,
                ),
            ),
            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(domain=sc_labels, range=sc_colors),
                legend=alt.Legend(
                    orient="bottom",
                    title=None,
                    labelFontSize=9,
                    symbolSize=60,
                    columns=2,
                ),
            ),
            tooltip=[
                "Scenario:N",
                "Metric:N",
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        )
        .properties(
            width=230,
            height=220,
            background="#FAFAFA",
            padding={"left": 4, "right": 4, "top": 4, "bottom": 4},
        )
        .configure_view(stroke=None)
        .configure_axis(grid=True)
        .configure(padding=0)
    )

    return bars


def chart_compare(ts_dumb, ts_smart, label_smart="C - MILP", color_smart=None,
                   title="Dumb Vs Smart: Total Depot Power"):
    """Side-by-side total kW lines with peak markers."""
    color_smart = color_smart or COL["smart"]
    df_d = ts_dumb[["hour", "P_total_kw"]].assign(scenario="A - Dumb")
    df_s = ts_smart[["hour", "P_total_kw"]].assign(scenario=label_smart)
    df = pd.concat([df_d, df_s], ignore_index=True)

    y_min = float(df["P_total_kw"].min())
    y_max = float(df["P_total_kw"].max())
    padding = (y_max - y_min) * 0.15 if (y_max - y_min) > 1e-6 else 5.0
    y_lo, y_hi = y_min - padding, y_max + padding

    lines = alt.Chart(df).mark_line(
        interpolate="step-after", strokeWidth=2.2
    ).encode(
        x=_hour_axis(),
        y=alt.Y("P_total_kw:Q", axis=alt.Axis(title="Total depot power (kW)"),
                scale=alt.Scale(domain=[y_lo, y_hi])),
        color=alt.Color("scenario:N",
                        scale=alt.Scale(domain=["A - Dumb", label_smart],
                                        range=[COL["dumb"], color_smart]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["scenario:N",
                 alt.Tooltip("P_total_kw:Q", format=".1f", title="kW"),
                 alt.Tooltip("hour:Q", title="Hour")],
    )

    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#888", strokeDash=[3, 2], opacity=0.5
    ).encode(y="y:Q")

    peak_rows = []
    for label, ts_x in [("A - Dumb", ts_dumb), (label_smart, ts_smart)]:
        i = int(ts_x["P_total_kw"].idxmax())
        peak_rows.append({"hour": float(ts_x.at[i, "hour"]),
                          "P_total_kw": float(ts_x.at[i, "P_total_kw"]),
                          "scenario": label})
    peaks = alt.Chart(pd.DataFrame(peak_rows)).mark_point(
        filled=True, size=120, shape="diamond"
    ).encode(
        x="hour:Q",
        y=alt.Y("P_total_kw:Q", scale=alt.Scale(domain=[y_lo, y_hi])),
        color=alt.Color("scenario:N",
                        scale=alt.Scale(domain=["A - Dumb", label_smart],
                                        range=[COL["dumb"], color_smart]),
                        legend=None),
        tooltip=["scenario:N",
                 alt.Tooltip("P_total_kw:Q", format=".1f", title="Peak kW"),
                 alt.Tooltip("hour:Q", title="Peak hour")],
    )

    return (lines + zero + peaks).properties(
        title=title, height=300, background="#FAFAFA"
    ).interactive()


def kpi_table_html(kpis_dict, comp_dict=None, title="Grid impact KPIs"):

    cols = [
        ("peak_total_kw",        "Peak (kW)",           ".1f"),
        ("energy_imported_kwh",  "Imported (kWh)",      ".1f"),
        ("energy_exported_kwh",  "Exported (kWh)",      ".1f"),
        ("v_min_pu",             "V min (p.u.)",        ".4f"),
        ("v_max_pu",             "V max (p.u.)",        ".4f"),
        ("v_violations_low",     "V viol. low (h)",     "d"),
        ("v_violations_high",    "V viol. high (h)",    "d"),
        ("trafo_max_loading",    "Trafo max (%)",       ".1f"),
        ("trafo_overload_hours", "Trafo overload (h)",  "d"),
    ]

    head_cells = "".join(f"<th>{lbl}</th>" for _, lbl, _ in cols)
    body_rows  = ""
    for sc_label, kpi in kpis_dict.items():
        cells = "".join(
            f"<td>{format(kpi[k], fmt)}</td>" for k, _, fmt in cols
        )
        body_rows += (
            f"<tr><td style='text-align:left;font-weight:600'>{sc_label}</td>"
            f"{cells}</tr>"
        )

    footer = ""
    if comp_dict is not None:
        footer = (
            f"<div>"
            f"</div>"
        )

    return f"""
<style>
.grid_kpi {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:6px; }}
.grid_kpi th, .grid_kpi td {{
    border:1px solid #d0d0d0; padding:5px 10px;
    text-align:center; white-space:nowrap;
}}
.grid_kpi th {{ background:#f0f2f6; font-weight:600; }}
.grid_kpi tbody tr:hover td {{ background:#eef4ff; }}
</style>
<b>{title}</b>
<table class='grid_kpi'>
<thead><tr><th style='text-align:left'>Scenario</th>{head_cells}</tr></thead>
<tbody>{body_rows}</tbody>
</table>
{footer}
"""

if __name__ == "__main__":
    from grid_sim import (
        build_grid, bdew_profile, run_grid_timeseries, extract_kpis, compare_scenarios,
        ANNUAL_OFFICE_KWH_DEFAULT, ANNUAL_ADMIN_KWH_DEFAULT, ANNUAL_INDUSTRIAL_KWH_DEFAULT,
    )

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

    P_office     = bdew_profile(ANNUAL_OFFICE_KWH_DEFAULT, slp_code="g1")
    P_admin      = bdew_profile(ANNUAL_ADMIN_KWH_DEFAULT, slp_code="g0")
    P_industrial = bdew_profile(ANNUAL_INDUSTRIAL_KWH_DEFAULT, slp_code="g3")

    ts_d = run_grid_timeseries(build_grid(), P_dumb, P_office, P_admin, P_industrial, n_trailers=10)
    ts_s = run_grid_timeseries(build_grid(), P_smart, P_office, P_admin, P_industrial, n_trailers=10)

    chart_power_stack(ts_d).save("chart1_power_dumb.html")
    chart_voltage(ts_d).save("chart2_voltage_dumb.html")
    chart_trafo_loading(ts_d).save("chart3_trafo_dumb.html")
    chart_compare(ts_d, ts_s).save("chart4_compare.html")
    print("All four charts saved to chart{1..4}_*.html")

    kpis = {"A - Dumb": extract_kpis(ts_d), "C - MILP": extract_kpis(ts_s)}
    comp = compare_scenarios(ts_d, ts_s)
    with open("grid_kpi.html", "w") as f:
        f.write("<html><body>" + kpi_table_html(kpis, comp) + "</body></html>")
    print("KPI table saved to grid_kpi.html")
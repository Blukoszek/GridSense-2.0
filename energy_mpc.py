
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy MPC Optimizer (5-min rolling horizon) for household PV + multi-battery + multi-inverter.
- Builds a linear program with PuLP each step over a horizon (default 24h) and dispatches the first 5-min action.
- Outputs results.csv, summary.json, validation_report.md and simple plots.

USAGE (example):
  python energy_mpc.py --csv input.csv --config config.json --outdir out --horizon-hours 24 --step-min 5 --validate

Input CSV columns (UTC+10: Australia/Brisbane assumed unless you pass --tz):
  timestamp,L_kW,PV_avail_kW,price_buy,price_sell

Notes:
- If price_sell missing, we use price_buy for both directions by default.
- If PuLP / a solver isn't installed, we'll warn with install tips.
"""
import argparse
import os
import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd

# Optional libs for plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to import PuLP and pick a solver
pulp_available = True
try:
    import pulp
except Exception as e:
    pulp_available = False

# ---------- Data classes ----------

@dataclass
class BatteryBank:
    name: str
    capacity_kWh: float
    soc_min: float
    soc_max: float
    p_charge_max_kW: float
    p_discharge_max_kW: float
    eta_ch: float = 0.97
    eta_dis: float = 0.97
    soc_init_kWh: float = None  # if None => set to midpoint at runtime


@dataclass
class Inverter:
    name: str
    charge_cap_kW: float
    discharge_cap_kW: float
    banks: List[str]  # names of banks connected to this inverter


@dataclass
class SiteConfig:
    import_limit_kW: float
    export_limit_kW: float
    allow_grid_charge: bool = True
    allow_export: bool = True


@dataclass
class Economics:
    deg_cost_per_kWh: float = 0.05
    epsilon_penalty: float = 1e-6


@dataclass
class Reserve:
    enabled: bool = False
    backup_hours: float = 0.0
    critical_load_kW: float = 0.0


@dataclass
class Config:
    timebase_min: int = 5
    horizon_hours: int = 24
    site: SiteConfig = None
    battery_banks: List[BatteryBank] = field(default_factory=list)
    inverters: List[Inverter] = field(default_factory=list)
    economics: Economics = Economics()
    reserve: Reserve = Reserve()


# ---------- Helpers ----------

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = json.load(f)
    site = SiteConfig(**d["site"])
    banks = [BatteryBank(**b) for b in d["battery_banks"]]
    invs = [Inverter(**inv) for inv in d["inverters"]]
    econ = Economics(**d.get("economics", {}))
    res = Reserve(**d.get("reserve", {}))
    return Config(
        timebase_min=d.get("timebase_min", 5),
        horizon_hours=d.get("horizon_hours", 24),
        site=site,
        battery_banks=banks,
        inverters=invs,
        economics=econ,
        reserve=res
    )


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    required = ["timestamp", "L_kW", "PV_avail_kW"]
    for c in required:
        if c not in cols:
            raise ValueError(f"Missing required column: {c}")
    if "price_buy" not in cols and "price_sell" not in cols and "price" not in cols:
        raise ValueError("Provide price columns: price_buy and price_sell (or a single 'price').")
    if "price_buy" not in cols and "price" in cols:
        df["price_buy"] = df["price"]
    if "price_sell" not in cols and "price" in cols:
        df["price_sell"] = df["price"]
    if "price_buy" not in cols and "price_sell" in cols:
        df["price_buy"] = df["price_sell"]
    if "price_sell" not in cols and "price_buy" in cols:
        df["price_sell"] = df["price_buy"]
    return df


def localize_and_check_freq(df: pd.DataFrame, tz: str, step_min: int) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(tz)  # assume local times in tz
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    df = df.sort_values("timestamp").set_index("timestamp")
    # Check uniform step
    diffs = df.index.to_series().diff().dropna().value_counts()
    if len(diffs) > 1 or diffs.index[0] != pd.Timedelta(minutes=step_min):
        raise ValueError(f"Time step not uniform {step_min} minutes. Found diffs: {diffs.head()}")
    return df


def pick_solver():
    if not pulp_available:
        return None, "PuLP not installed. Try: pip install pulp"
    # Prefer CBC if available
    try:
        solver = pulp.PULP_CBC_CMD(msg=False)
        return solver, "Using PuLP CBC solver."
    except Exception:
        pass
    # Try default solver
    try:
        solver = pulp.getSolver("PULP_CBC_CMD", msg=False)
        return solver, "Using default PuLP solver."
    except Exception:
        pass
    return None, "No PuLP solver found. Install CBC or GLPK. e.g.: pip install pulp && apt-get install -y coinor-cbc"


# ---------- LP builder ----------

def build_lp_window(window: pd.DataFrame, soc0: Dict[str, float], cfg: Config, dt_hours: float):
    """
    Build a PuLP LP for the window and return (model, variables dict).
    variables is a dict keyed by names -> list/arrays of pulp variables over s steps.
    """
    S = len(window)
    banks = cfg.battery_banks
    site = cfg.site
    econ = cfg.economics

    if not pulp_available:
        raise RuntimeError("PuLP not installed. Cannot build LP.")

    model = pulp.LpProblem("MPC_Window", pulp.LpMaximize)

    # Variables per step
    P_grid_in   = [pulp.LpVariable(f"P_grid_in_{s}", lowBound=0) for s in range(S)]
    P_grid_out  = [pulp.LpVariable(f"P_grid_out_{s}", lowBound=0) for s in range(S)]
    P_pv_load   = [pulp.LpVariable(f"P_pv_to_load_{s}", lowBound=0) for s in range(S)]
    P_pv_batt   = [pulp.LpVariable(f"P_pv_to_batt_{s}", lowBound=0) for s in range(S)]
    P_pv_grid   = [pulp.LpVariable(f"P_pv_to_grid_{s}", lowBound=0) for s in range(S)]
    P_pv_curt   = [pulp.LpVariable(f"P_pv_curt_{s}", lowBound=0) for s in range(S)]

    P_batt_ch = {b.name: [pulp.LpVariable(f"P_ch_{b.name}_{s}", lowBound=0) for s in range(S)] for b in banks}
    P_batt_dis= {b.name: [pulp.LpVariable(f"P_dis_{b.name}_{s}", lowBound=0) for s in range(S)] for b in banks}
    SoC       = {b.name: [pulp.LpVariable(f"SoC_{b.name}_{s}", lowBound=b.soc_min, upBound=b.soc_max) for s in range(S)] for b in banks}

    # Battery dynamics & power bounds
    for s in range(S):
        for b in banks:
            model += P_batt_ch[b.name][s] <= b.p_charge_max_kW, f"ch_cap_{b.name}_{s}"
            model += P_batt_dis[b.name][s] <= b.p_discharge_max_kW, f"dis_cap_{b.name}_{s}"

    # SoC dynamics
    for b in banks:
        # Initialize SoC at step 0
        model += SoC[b.name][0] == soc0[b.name], f"soc_init_{b.name}"
        for s in range(S-1):
            ch_term = b.eta_ch * P_batt_ch[b.name][s] * dt_hours
            dis_term= (1.0/b.eta_dis) * P_batt_dis[b.name][s] * dt_hours
            model += SoC[b.name][s+1] == SoC[b.name][s] + ch_term - dis_term, f"soc_dyn_{b.name}_{s}"

    # Inverter limits per step
    for s in range(S):
        # Total bank charge/discharge hitting each inverter
        for inv in cfg.inverters:
            sum_ch = sum(P_batt_ch[bname][s] for bname in inv.banks)
            sum_dis= sum(P_batt_dis[bname][s] for bname in inv.banks)
            model += sum_ch <= inv.charge_cap_kW, f"inv_ch_{inv.name}_{s}"
            model += sum_dis <= inv.discharge_cap_kW, f"inv_dis_{inv.name}_{s}"

        # Site import/export caps
        model += P_grid_in[s]  <= site.import_limit_kW, f"site_imp_{s}"
        model += P_grid_out[s] <= site.export_limit_kW, f"site_exp_{s}"

        # PV availability
        pv_avail = float(window["PV_avail_kW"].iloc[s])
        model += (P_pv_load[s] + P_pv_batt[s] + P_pv_grid[s] + P_pv_curt[s]) <= pv_avail, f"pv_avail_{s}"

        # Policy toggles
        if not site.allow_export:
            model += P_grid_out[s] == 0, f"no_export_{s}"

        # Node balance:
        # PV to load/batt/grid + grid_in + batt_dis = load + batt_ch + grid_out + pv_curt
        L = float(window["L_kW"].iloc[s])
        lhs = P_pv_load[s] + P_pv_batt[s] + P_pv_grid[s] + P_grid_in[s] + sum(P_batt_dis[b.name][s] for b in banks)
        rhs = L + sum(P_batt_ch[b.name][s] for b in banks) + P_grid_out[s] + P_pv_curt[s]
        model += lhs == rhs, f"node_balance_{s}"

        # Optional: forbid direct grid->battery charging if policy says so
        if not site.allow_grid_charge:
            model += P_pv_batt[s] >= sum(P_batt_ch[b.name][s] for b in banks), f"pv_only_batt_{s}"

    # Objective: profit - cost - degradation - epsilon penalty
    obj_terms = []
    for s in range(S):
        buy  = float(window["price_buy"].iloc[s])
        sell = float(window["price_sell"].iloc[s])
        dt = dt_hours
        obj_terms.append(+ sell * P_grid_out[s] * dt)            # revenue
        obj_terms.append(- buy  * P_grid_in[s]  * dt)            # energy cost
        deg = sum(P_batt_ch[b.name][s] + P_batt_dis[b.name][s] for b in banks)
        obj_terms.append(- cfg.economics.deg_cost_per_kWh * deg * dt) # degradation
        obj_terms.append(- cfg.economics.epsilon_penalty * (P_grid_in[s] + P_grid_out[s]) * dt) # tie-break

    model += pulp.lpSum(obj_terms)

    variables = dict(
        P_grid_in=P_grid_in,
        P_grid_out=P_grid_out,
        P_pv_load=P_pv_load,
        P_pv_batt=P_pv_batt,
        P_pv_grid=P_pv_grid,
        P_pv_curt=P_pv_curt,
        P_batt_ch=P_batt_ch,
        P_batt_dis=P_batt_dis,
        SoC=SoC
    )
    return model, variables


def extract_first_step(variables: Dict[str, Any], banks: List[BatteryBank], default0: float = 0.0) -> Dict[str, float]:
    def val(v):
        try:
            x = v.value()
            return float(x) if x is not None else default0
        except Exception:
            return default0
    out = {}
    out["P_grid_in"]  = val(variables["P_grid_in"][0])
    out["P_grid_out"] = val(variables["P_grid_out"][0])
    out["P_pv_to_load"] = val(variables["P_pv_load"][0])
    out["P_pv_to_batt"] = val(variables["P_pv_batt"][0])
    out["P_pv_to_grid"] = val(variables["P_pv_grid"][0])
    out["P_pv_curt"]    = val(variables["P_pv_curt"][0])
    for b in banks:
        out[f"P_ch__{b.name}"]  = val(variables["P_batt_ch"][b.name][0])
        out[f"P_dis__{b.name}"] = val(variables["P_batt_dis"][b.name][0])
        out[f"SoC__{b.name}"]   = val(variables["SoC"][b.name][0])
    return out


def run_mpc(df: pd.DataFrame, cfg: Config, outdir: str, tz: str, horizon_hours: int=None, verbose: bool=True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if horizon_hours is None:
        horizon_hours = cfg.horizon_hours
    dt_hours = cfg.timebase_min / 60.0
    steps_per_hour = int(round(1.0 / dt_hours))
    H = horizon_hours * steps_per_hour

    # Initial SoC for each bank
    soc_now = {}
    for b in cfg.battery_banks:
        if b.soc_init_kWh is None:
            mid = 0.5 * (b.soc_min + b.soc_max)
            soc_now[b.name] = mid
        else:
            soc_now[b.name] = b.soc_init_kWh

    # Accumulators for results
    rows = []

    if not pulp_available:
        raise RuntimeError("PuLP is not installed. Install with: pip install pulp")

    solver, solver_msg = pick_solver()
    if solver is None:
        raise RuntimeError("No suitable LP solver found. " + solver_msg)

    if verbose:
        print(solver_msg)

    for t0_idx, (ts, row) in enumerate(df.iterrows()):
        window = df.iloc[t0_idx : t0_idx + H]
        if len(window) < H:
            break

        model, variables = build_lp_window(window, soc_now, cfg, dt_hours)

        status = model.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            print(f"[{ts}] Warning: LP status = {pulp.LpStatus[status]}")

        step = extract_first_step(variables, cfg.battery_banks)

        for b in cfg.battery_banks:
            ch = step[f"P_ch__{b.name}"]
            dis = step[f"P_dis__{b.name}"]
            soc_now[b.name] = max(b.soc_min, min(b.soc_max, soc_now[b.name] + (b.eta_ch*ch - (1.0/b.eta_dis)*dis) * dt_hours))

        rec = {"timestamp": ts}
        rec.update(step)
        rec["L_kW"] = float(row["L_kW"])
        rec["PV_avail_kW"] = float(row["PV_avail_kW"])
        rec["price_buy"] = float(row["price_buy"])
        rec["price_sell"] = float(row["price_sell"])
        rows.append(rec)

    res = pd.DataFrame(rows).set_index("timestamp")

    kpis = compute_kpis(res, cfg, dt_hours)

    os.makedirs(outdir, exist_ok=True)
    res.to_csv(os.path.join(outdir, "results.csv"))
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(kpis, f, indent=2)

    val_report = validation_report(df.loc[res.index], res, cfg, dt_hours)
    with open(os.path.join(outdir, "validation_report.md"), "w") as f:
        f.write(val_report)

    plot_outputs(res, outdir)

    return res, kpis


def compute_kpis(res: pd.DataFrame, cfg: Config, dt_hours: float) -> Dict[str, Any]:
    E_import = (res["P_grid_in"] * dt_hours).sum()
    E_export = (res["P_grid_out"] * dt_hours).sum()

    E_pv_load = (res["P_pv_to_load"] * dt_hours).sum()
    E_pv_batt = (res["P_pv_to_batt"] * dt_hours).sum()
    E_pv_grid = (res["P_pv_to_grid"] * dt_hours).sum()
    E_pv_curt = (res["P_pv_curt"] * dt_hours).sum()

    batt_throughput = 0.0
    for b in cfg.battery_banks:
        batt_throughput += ((res[f"P_ch__{b.name}"] + res[f"P_dis__{b.name}"]) * dt_hours).sum()

    revenue = (res["P_grid_out"] * res["price_sell"] * dt_hours).sum()
    cost    = (res["P_grid_in"]  * res["price_buy"]  * dt_hours).sum()
    deg_cost= cfg.economics.deg_cost_per_kWh * batt_throughput
    profit  = revenue - cost - deg_cost

    kpis = dict(
        energy_import_kWh = E_import,
        energy_export_kWh = E_export,
        pv_to_load_kWh = E_pv_load,
        pv_to_batt_kWh = E_pv_batt,
        pv_to_grid_kWh = E_pv_grid,
        pv_curtail_kWh = E_pv_curt,
        batt_throughput_kWh = batt_throughput,
        revenue_$ = revenue,
        import_cost_$ = cost,
        degradation_cost_$ = deg_cost,
        profit_$ = profit,
        time_steps = len(res)
    )
    return kpis


def validation_report(df_in: pd.DataFrame, res: pd.DataFrame, cfg: Config, dt_hours: float) -> str:
    lines = []
    add = lines.append
    add("# Validation Report\n")
    add(f"- Steps: {len(res)}")
    add(f"- Δt (hours): {dt_hours:.5f}\n")

    residuals = []
    for ts, row in res.iterrows():
        lhs = row["P_pv_to_load"] + row["P_pv_to_batt"] + row["P_pv_to_grid"] + row["P_grid_in"]
        for b in cfg.battery_banks:
            lhs += row[f"P_dis__{b.name}"]
        rhs = df_in.loc[ts, "L_kW"]
        for b in cfg.battery_banks:
            rhs += row[f"P_ch__{b.name}"]
        rhs += row["P_grid_out"] + row["P_pv_curt"]
        residuals.append(lhs - rhs)
    res_series = pd.Series(residuals, index=res.index)
    max_abs_resid = res_series.abs().max()
    add(f"## Energy Node Balance\n- Max |residual| (kW): {max_abs_resid:.6f} (target ≤ 1e-6)\n")

    imp_viol = (res["P_grid_in"]  > cfg.site.import_limit_kW + 1e-9).sum()
    exp_viol = (res["P_grid_out"] > cfg.site.export_limit_kW + 1e-9).sum()

    pv_feed = res["P_pv_to_load"] + res["P_pv_to_batt"] + res["P_pv_to_grid"] + res["P_pv_curt"]
    pv_viol = (pv_feed > (df_in["PV_avail_kW"] + 1e-9)).sum()

    inv_viol = 0
    for inv in cfg.inverters:
        sum_ch = res[[f"P_ch__{b}" for b in inv.banks]].sum(axis=1)
        sum_dis= res[[f"P_dis__{b}" for b in inv.banks]].sum(axis=1)
        inv_viol += (sum_ch > inv.charge_cap_kW + 1e-9).sum()
        inv_viol += (sum_dis > inv.discharge_cap_kW + 1e-9).sum()

    add("## Limits\n"
        f"- Import cap violations: {imp_viol}\n"
        f"- Export cap violations: {exp_viol}\n"
        f"- PV availability violations: {pv_viol}\n"
        f"- Inverter cap violations (charge+discharge counted): {inv_viol}\n")

    soc_viol = 0
    for b in cfg.battery_banks:
        s = res[f"SoC__{b.name}"]
        soc_viol += ((s < b.soc_min - 1e-9) | (s > b.soc_max + 1e-9)).sum()
    add(f"## State of Charge Bounds\n- SoC violations: {soc_viol}\n")

    both = ((res["P_grid_in"] > 1e-6) & (res["P_grid_out"] > 1e-6)).sum()
    add(f"## Simultaneous Import & Export\n- Steps with both > 0: {both}\n")

    neg_buy_steps = (df_in["price_buy"] < 0).sum()
    add(f"## Price Sanity\n- Steps with negative buy price: {neg_buy_steps}\n")

    kpis = compute_kpis(res, cfg, dt_hours)
    add("## KPIs\n")
    for k, v in kpis.items():
        if isinstance(v, (int,)):
            add(f"- {k}: {v}")
        else:
            add(f"- {k}: {v:.6f}" if isinstance(v, float) else f"- {k}: {v}")
    add("")
    return "\n".join(lines)


def plot_outputs(res: pd.DataFrame, outdir: str):
    plt.figure()
    for col in [c for c in res.columns if c.startswith("SoC__")]:
        res[col].plot()
    plt.title("State of Charge (kWh)")
    plt.xlabel("Time")
    plt.ylabel("kWh")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_soc.png"))
    plt.close()

    plt.figure()
    res["P_grid_in"].plot(label="Grid Import (kW)")
    res["P_grid_out"].plot(label="Grid Export (kW)")
    plt.legend()
    plt.title("Grid Import/Export (kW)")
    plt.xlabel("Time")
    plt.ylabel("kW")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_grid_flows.png"))
    plt.close()

    plt.figure()
    res["P_pv_to_load"].plot(label="PV→Load")
    res["P_pv_to_batt"].plot(label="PV→Batt")
    res["P_pv_to_grid"].plot(label="PV→Grid")
    res["P_pv_curt"].plot(label="PV Curtailment")
    plt.legend()
    plt.title("PV Allocation (kW)")
    plt.xlabel("Time")
    plt.ylabel("kW")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_pv_alloc.png"))
    plt.close()

    for bname in sorted({c.split("__")[-1] for c in res.columns if c.startswith("P_ch__")}):
        plt.figure()
        res[f"P_ch__{bname}"].plot(label=f"{bname} Charge")
        res[f"P_dis__{bname}"].plot(label=f"{bname} Discharge")
        plt.legend()
        plt.title(f"Battery Flows — {bname} (kW)")
        plt.xlabel("Time")
        plt.ylabel("kW")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"plot_battery_{bname}.png"))
        plt.close()


def make_arg_parser():
    ap = argparse.ArgumentParser(description="Energy MPC optimizer for PV + battery household")
    ap.add_argument("--csv", required=False, help="Path to input CSV with timestamp,L_kW,PV_avail_kW,price_buy,price_sell")
    ap.add_argument("--config", required=False, default="config_default.json", help="Path to config JSON")
    ap.add_argument("--outdir", required=False, default="out", help="Output directory")
    ap.add_argument("--tz", required=False, default="Australia/Brisbane", help="Timezone for timestamps")
    ap.add_argument("--horizon-hours", type=int, default=None, help="Override config horizon hours")
    ap.add_argument("--step-min", type=int, default=None, help="Override config timebase (minutes)")
    ap.add_argument("--validate", action="store_true", help="Run validation and write validation_report.md")
    ap.add_argument("--generate-template", action="store_true", help="Write an input_template.csv next to script and exit")
    return ap


def write_template_csv(path: str, tz: str = "Australia/Brisbane"):
    import pandas as pd, numpy as np
    idx = pd.date_range("2025-08-11 00:00", periods=288, freq="5min", tz=tz)
    hours = idx.tz_convert("UTC").tz_localize(None).hour.values
    base = 1.0 + 0.8 * np.sin((hours-7)/24*2*np.pi).clip(-1, 1)
    L_kW = base + 0.5*np.random.rand(len(idx))
    t = np.arange(len(idx))
    pv = 12.0 * np.exp(-0.5*((t - len(idx)/2)/(len(idx)/6))**2)
    pv *= (hours>=6) & (hours<=18)
    price = 0.20 + 0.05*np.sin(t/50.0) + 0.10*(np.random.rand(len(idx))>0.98)
    df = pd.DataFrame({
        "timestamp": idx,
        "L_kW": L_kW,
        "PV_avail_kW": pv,
        "price_buy": price,
        "price_sell": price - 0.02
    })
    df.to_csv(path, index=False)


def main():
    ap = make_arg_parser()
    args = ap.parse_args()

    if args.generate_template:
        outp = os.path.join(os.path.dirname(__file__), "input_template.csv")
        write_template_csv(outp, tz=args.tz)
        print(f"Wrote template to {outp}")
        return

    cfg = load_config(args.config)
    if args.step_min is not None:
        cfg.timebase_min = args.step_min
    if args.horizon_hours is not None:
        cfg.horizon_hours = args.horizon_hours

    if args.csv is None:
        raise SystemExit("Please provide --csv path to input data. Or run with --generate-template to create one.")

    df = pd.read_csv(args.csv)
    df = ensure_columns(df)
    df = localize_and_check_freq(df, args.tz, cfg.timebase_min)

    res, kpis = run_mpc(df, cfg, args.outdir, args.tz, horizon_hours=cfg.horizon_hours)
    print("Done. Outputs in:", args.outdir)
    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()

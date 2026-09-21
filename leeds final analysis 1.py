

import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from wildboottest.wildboottest import wildboottest  
LEEDS_AADF = "leeds_aadf.csv"
BRADFORD_AADF = "bradford_aadf.csv"

BRADFORD_BOUNDARY_POINT = 73114
CS1_YEAR = 2016           
CONSTRUCTION_START = 2020  
COMPLETION_YEAR = 2022     
HAC_LAGS = 2
PRETREND_MIN_YEARS = 15   
SIGNIFICANCE = 0.05

CANDIDATE_LEEDS_ROADS = ["A61", "A64", "A65", "A657", "A660"]


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)



section("SETUP: Loading local files")

for path, label in [(LEEDS_AADF, "Leeds"), (BRADFORD_AADF, "Bradford")]:
    if not os.path.exists(path):
        raise SystemExit(f"\nERROR: {label} file not found at:\n{path}\nUpdate the path and rerun.")

leeds = pd.read_csv(LEEDS_AADF, low_memory=False)
bradford = pd.read_csv(BRADFORD_AADF, low_memory=False)

for df, label in [(leeds, "Leeds"), (bradford, "Bradford")]:
    if "hour" in df.columns:
        raise SystemExit(f"\nERROR: {label} file is RAW COUNTS (has 'hour' column), not AADF.")

print(f"Leeds:    {leeds.shape[0]} rows, {leeds['count_point_id'].nunique()} count points")
print(f"Bradford: {bradford.shape[0]} rows, {bradford['count_point_id'].nunique()} count points")

a647_leeds = leeds[leeds["road_name"] == "A647"]
a647_bradford_all = bradford[bradford["road_name"] == "A647"]




section("SECTION A: Data validation")

print(f"A647 (Leeds) count points: {a647_leeds['count_point_id'].nunique()}")
detail = a647_leeds.groupby("count_point_id").agg(
    start_junction=("start_junction_road_name", "first"),
    end_junction=("end_junction_road_name", "first"),
    longitude=("longitude", "first"),
    n_years=("year", "nunique"),
).sort_values("longitude")
print(detail.to_string())

a647_leeds_yearly = a647_leeds.groupby("year")["all_motor_vehicles"].mean().rename("value")



section("SECTION B: Comparator screening - pre-2016 pre-trend test (verified methodology)")


def pretrend_test(treated_yearly, comparator_yearly, label):
    """treated_yearly and comparator_yearly are Series indexed by year."""
    pre_t = treated_yearly[treated_yearly.index < CS1_YEAR].rename("value").reset_index()
    pre_t["treated"] = 1
    pre_c = comparator_yearly[comparator_yearly.index < CS1_YEAR].rename("value").reset_index()
    pre_c["treated"] = 0
    test_df = pd.concat([pre_t, pre_c], ignore_index=True).dropna()
    if test_df["treated"].nunique() < 2 or len(test_df) < 6:
        return {"comparator": label, "n_obs": len(test_df), "slope_diff": None, "p_value": None, "result": "insufficient data"}
    test_df["year_c"] = test_df["year"] - test_df["year"].min()
    m = smf.ols("value ~ year_c * treated", data=test_df).fit()
    coef = m.params.get("year_c:treated", float("nan"))
    pval = m.pvalues.get("year_c:treated", float("nan"))
    result = "PASS (parallel)" if pval > SIGNIFICANCE else "FAIL (differs)"
    return {"comparator": label, "n_obs": len(test_df), "slope_diff": round(coef, 1),
            "p_value": round(pval, 3), "result": result}


screening = []

for road in CANDIDATE_LEEDS_ROADS:
    comp = leeds[leeds["road_name"] == road]
    n_points = comp["count_point_id"].nunique()
    comp_yearly = comp.groupby("year")["all_motor_vehicles"].mean()
    r = pretrend_test(a647_leeds_yearly, comp_yearly, road)
    r["n_count_points"] = n_points
    screening.append(r)

single = bradford[bradford["count_point_id"] == BRADFORD_BOUNDARY_POINT]
single_yearly = single.groupby("year")["all_motor_vehicles"].mean()
r = pretrend_test(a647_leeds_yearly, single_yearly, "Bradford single point (73114)")
r["n_count_points"] = 1
screening.append(r)

coverage = a647_bradford_all.groupby("count_point_id")["year"].nunique()
well_covered = coverage[coverage >= PRETREND_MIN_YEARS].index.tolist()
stretch = a647_bradford_all[a647_bradford_all["count_point_id"].isin(well_covered)]
stretch_yearly = stretch.groupby("year")["all_motor_vehicles"].mean()
r = pretrend_test(a647_leeds_yearly, stretch_yearly, "Bradford whole stretch (avg)")
r["n_count_points"] = len(well_covered)
screening.append(r)

screening_df = pd.DataFrame(screening)[["comparator", "n_count_points", "n_obs", "slope_diff", "p_value", "result"]]
print(screening_df.to_string(index=False))
screening_df.to_csv("comparator_screening_v2.csv", index=False)
print("\nSaved comparator_screening_v2.csv")

passing = screening_df[screening_df["result"] == "PASS (parallel)"].sort_values("n_count_points", ascending=False)
print(f"\nComparators that PASS the pre-trend test: {passing['comparator'].tolist()}")

if len(passing) == 0:
    print("\nWARNING: no comparator passes the pre-trend test. Consider synthetic control")
    print("(a weighted blend of candidates) rather than forcing a single comparator.")
    comparators_to_use = {}
else:
    comparators_to_use = {}
    for name in passing["comparator"]:
        if name == "Bradford single point (73114)":
            comparators_to_use[name] = single_yearly
        elif name == "Bradford whole stretch (avg)":
            comparators_to_use[name] = stretch_yearly
        else:
            comparators_to_use[name] = leeds[leeds["road_name"] == name].groupby("year")["all_motor_vehicles"].mean()
    print(f"\nCarrying these into Sections D-H: {list(comparators_to_use.keys())}")



def build_its_vars(df, lptip_year=CONSTRUCTION_START):
    df = df.copy()
    df["T"] = df["year"] - df["year"].min()
    df["D1"] = (df["year"] >= CS1_YEAR).astype(int)
    df["T1"] = (df["year"] - CS1_YEAR).clip(lower=0)
    df["D2"] = (df["year"] >= lptip_year).astype(int)
    df["T2"] = (df["year"] - lptip_year).clip(lower=0)
    return df



section("SECTION C: Two-breakpoint ITS (A647 alone)")

a647_df = a647_leeds_yearly.rename("all_motor_vehicles").reset_index()
for exclude_covid in [False, True]:
    df = a647_df[~a647_df["year"].isin([2020, 2021])] if exclude_covid else a647_df
    df = build_its_vars(df)
    m = smf.ols("all_motor_vehicles ~ T + D1 + T1 + D2 + T2", data=df).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    tag = "excl. COVID" if exclude_covid else "full period"
    print(f"\n--- {tag} ---")
    for term in ["T", "D1", "T1", "D2", "T2"]:
        print(f"  {term}: {m.params[term]:+9.1f}  (p={m.pvalues[term]:.3f})")



section("SECTION D: Comparative ITS / DiD vs every comparator that passed screening")

def run_did(comp_yearly, exclude_covid):
    a647_y = a647_leeds_yearly.rename("all_motor_vehicles").reset_index()
    a647_y["treated"] = 1
    comp_y = comp_yearly.rename("all_motor_vehicles").reset_index()
    comp_y["treated"] = 0
    combined = pd.concat([a647_y, comp_y], ignore_index=True)
    if exclude_covid:
        combined = combined[~combined["year"].isin([2020, 2021])]
    combined = build_its_vars(combined)
    return smf.ols("all_motor_vehicles ~ (T + D1 + T1 + D2 + T2) * treated", data=combined).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})

did_summary = []
for label, comp_yearly in comparators_to_use.items():
    for exclude_covid in [False, True]:
        m = run_did(comp_yearly, exclude_covid)
        tag = "excl_covid" if exclude_covid else "full"
        d2, d2p = m.params.get("D2:treated", float("nan")), m.pvalues.get("D2:treated", float("nan"))
        t2, t2p = m.params.get("T2:treated", float("nan")), m.pvalues.get("T2:treated", float("nan"))
        did_summary.append({"comparator": label, "spec": tag, "D2_treated": round(d2, 1), "D2_p": round(d2p, 3),
                              "T2_treated": round(t2, 1), "T2_p": round(t2p, 3)})
        print(f"{label:32s} [{tag:11s}]  D2:treated={d2:+9.1f} (p={d2p:.3f})   T2:treated={t2:+9.1f} (p={t2p:.3f})")

did_df = pd.DataFrame(did_summary)
did_df.to_csv("did_summary_v2.csv", index=False)
print("\nSaved did_summary_v2.csv")



if len(passing) > 0:
    primary_label = passing.iloc[0]["comparator"]
    primary_yearly = comparators_to_use[primary_label]

    section(f"SECTION E: Jackknife (drop one A647 point at a time) vs '{primary_label}', excl. COVID")
    for dropped in a647_leeds["count_point_id"].unique():
        subset = a647_leeds[a647_leeds["count_point_id"] != dropped]
        subset_yearly = subset.groupby("year")["all_motor_vehicles"].mean()
        a647_y = subset_yearly.rename("all_motor_vehicles").reset_index()
        a647_y["treated"] = 1
        comp_y = primary_yearly.rename("all_motor_vehicles").reset_index()
        comp_y["treated"] = 0
        combined = pd.concat([a647_y, comp_y], ignore_index=True)
        combined = combined[~combined["year"].isin([2020, 2021])]
        combined = build_its_vars(combined)
        m = smf.ols("all_motor_vehicles ~ (T + D1 + T1 + D2 + T2) * treated", data=combined).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        t2, t2p = m.params.get("T2:treated", float("nan")), m.pvalues.get("T2:treated", float("nan"))
        print(f"Dropping {dropped}: T2:treated={t2:+.1f} (p={t2p:.3f})")

    section(f"SECTION F: Break-year sensitivity vs '{primary_label}', excl. COVID")
    for year in [2019, 2020, 2021]:
        a647_y = a647_leeds_yearly.rename("all_motor_vehicles").reset_index()
        a647_y["treated"] = 1
        comp_y = primary_yearly.rename("all_motor_vehicles").reset_index()
        comp_y["treated"] = 0
        combined = pd.concat([a647_y, comp_y], ignore_index=True)
        combined = combined[~combined["year"].isin([2020, 2021])]
        combined = build_its_vars(combined, lptip_year=year)
        m = smf.ols("all_motor_vehicles ~ (T + D1 + T1 + D2 + T2) * treated", data=combined).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        t2, t2p = m.params.get("T2:treated", float("nan")), m.pvalues.get("T2:treated", float("nan"))
        print(f"Break year {year}: T2:treated={t2:+.1f} (p={t2p:.3f})")
else:
    primary_label, primary_yearly = None, None
    print("\nSkipping Sections E-H: no comparator passed screening.")



if primary_yearly is not None:
    section(f"SECTION G: Construction-phase (2020-2021) vs post-completion (2022+), vs '{primary_label}'")

    def build_split_vars(df):
        df = df.copy()
        df["T"] = df["year"] - df["year"].min()
        df["D1"] = (df["year"] >= CS1_YEAR).astype(int)
        df["T1"] = (df["year"] - CS1_YEAR).clip(lower=0)
        df["Dc"] = (df["year"] >= CONSTRUCTION_START).astype(int)
        df["Tc"] = (df["year"] - CONSTRUCTION_START).clip(lower=0)
        df["Dp"] = (df["year"] >= COMPLETION_YEAR).astype(int)
        df["Tp"] = (df["year"] - COMPLETION_YEAR).clip(lower=0)
        return df

    a647_y = a647_leeds_yearly.rename("all_motor_vehicles").reset_index()
    a647_y["treated"] = 1
    comp_y = primary_yearly.rename("all_motor_vehicles").reset_index()
    comp_y["treated"] = 0
    combined = pd.concat([a647_y, comp_y], ignore_index=True)
    combined = combined[~combined["year"].isin([2020, 2021])]
    combined = build_split_vars(combined)
    m = smf.ols("all_motor_vehicles ~ (T + D1 + T1 + Dc + Tc + Dp + Tp) * treated", data=combined).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    for term, desc in [("Dc:treated", "Level shift, construction start"), ("Tc:treated", "Trend, during construction"),
                         ("Dp:treated", "Level shift, post-completion"), ("Tp:treated", "Trend, post-completion")]:
        if term in m.params.index:
            print(f"  {desc:35s}: {m.params[term]:+9.1f}  (p={m.pvalues[term]:.3f})")



if primary_yearly is not None:
    section(f"SECTION H: Bus vs car volume vs '{primary_label}', full period and excl. COVID")
    for outcome in ["buses_and_coaches", "cars_and_taxis"]:
        a647_y = a647_leeds.groupby("year")[outcome].mean().rename("value").reset_index()
        a647_y["treated"] = 1
        if primary_label == "Bradford single point (73114)":
            comp_src = bradford[bradford["count_point_id"] == BRADFORD_BOUNDARY_POINT]
        elif primary_label == "Bradford whole stretch (avg)":
            comp_src = stretch
        else:
            comp_src = leeds[leeds["road_name"] == primary_label]
        comp_y = comp_src.groupby("year")[outcome].mean().rename("value").reset_index()
        comp_y["treated"] = 0
        combined_all = pd.concat([a647_y, comp_y], ignore_index=True)
        combined_all = combined_all.rename(columns={"value": outcome})

        for spec_label, spec_df in [("full", combined_all), ("excl_covid", combined_all[~combined_all["year"].isin([2020, 2021])])]:
            spec_df = build_its_vars(spec_df)
            m = smf.ols(f"{outcome} ~ (T + D1 + T1 + D2 + T2) * treated", data=spec_df).fit(
                cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            d2, d2p = m.params.get("D2:treated", float("nan")), m.pvalues.get("D2:treated", float("nan"))
            t2, t2p = m.params.get("T2:treated", float("nan")), m.pvalues.get("T2:treated", float("nan"))
            print(f"{outcome:20s} [{spec_label:11s}]: D2:treated={d2:+.2f} (p={d2p:.3f})   T2:treated={t2:+.2f} (p={t2p:.3f})")


section("SECTION I: Summary")
print(f"""
Comparators screened: {len(screening_df)}
Comparators that passed pre-trend test: {passing['comparator'].tolist() if len(passing) > 0 else 'NONE'}
Primary comparator used for Sections E-H: {primary_label if primary_yearly is not None else 'N/A - none passed'}

Full detail saved to: comparator_screening_v2.csv, did_summary_v2.csv
""")



section("SECTION J: Genuine panel-level DiD (clustered by count_point_id) + wild-cluster bootstrap")

if primary_yearly is None:
    print("Skipping Section J: no comparator passed screening.")
else:
   
    if primary_label == "Bradford single point (73114)":
        comp_raw = bradford[bradford["count_point_id"] == BRADFORD_BOUNDARY_POINT].copy()
    elif primary_label == "Bradford whole stretch (avg)":
        comp_raw = stretch.copy()
    else:
        comp_raw = leeds[leeds["road_name"] == primary_label].copy()

    n_comp_points = comp_raw["count_point_id"].nunique()
    n_a647_points = a647_leeds["count_point_id"].nunique()
    print(f"Primary comparator: {primary_label}")
    print(f"  A647 count points (clusters): {n_a647_points}")
    print(f"  Comparator count points (clusters): {n_comp_points}")

    if n_comp_points < 2:
        print(
            "\nWARNING: comparator has only 1 count point, giving very few total "
            "clusters. Cluster-robust inference (and the wild-cluster bootstrap) "
            "is known to be unreliable with this few clusters (Cameron, Gelbach "
            "and Miller, 2008) - treat results here as indicative only, and "
            "consider Section D's HAC-based results as the more defensible "
            "primary specification in that case."
        )

  
    a647_panel = a647_leeds[["count_point_id", "year", "all_motor_vehicles"]].copy()
    a647_panel["treated"] = 1
    comp_panel = comp_raw[["count_point_id", "year", "all_motor_vehicles"]].copy()
    comp_panel["treated"] = 0

    panel = pd.concat([a647_panel, comp_panel], ignore_index=True)
    panel = panel[~panel["year"].isin([2020, 2021])] 
    panel = build_its_vars(panel)

    print(f"\nPanel shape: {panel.shape}  (rows = count_point_id x year combinations)")
    print(f"Total clusters (unique count_point_id): {panel['count_point_id'].nunique()}")

 
    ols_model = smf.ols("all_motor_vehicles ~ (T + D1 + T1 + D2 + T2) * treated", data=panel)
    fit = ols_model.fit(cov_type="cluster", cov_kwds={"groups": panel["count_point_id"]})

    print("\n--- Panel DiD, cluster-robust SEs ---")
    for term in ["D2:treated", "T2:treated"]:
        coef, p = fit.params.get(term, float("nan")), fit.pvalues.get(term, float("nan"))
        print(f"  {term}: {coef:+9.1f}  (p={p:.3f})")

    print("\n--- Wild-cluster bootstrap (Rademacher weights, 9999 reps) ---")
    for param in ["D2:treated", "T2:treated"]:
        try:
            result = wildboottest(
                ols_model,
                cluster=panel["count_point_id"],
                param=param,
                bootstrap_type="11",
                B=9999,
            )
            print(f"\n{param}:")
            print(result)
        except KeyError:
            print(f"\n{param}: KeyError - check fit.summary() below for the exact term name.")
            print(fit.summary())

print("\nPipeline complete (Sections A-J).")



section("SECTION K: Robustness on buses_and_coaches specifically, vs A61")

if primary_yearly is None:
    print("Skipping Section K: no comparator passed screening.")
else:
    OUTCOME = "buses_and_coaches"

    section(f"SECTION K1: Jackknife on '{OUTCOME}' (drop one A647 point at a time) vs 'A61', excl. COVID")
    comp_src_bus = leeds[leeds["road_name"] == "A61"]
    comp_bus_yearly = comp_src_bus.groupby("year")[OUTCOME].mean()
    for dropped in a647_leeds["count_point_id"].unique():
        subset = a647_leeds[a647_leeds["count_point_id"] != dropped]
        subset_yearly = subset.groupby("year")[OUTCOME].mean().rename(OUTCOME).reset_index()
        subset_yearly["treated"] = 1
        comp_y = comp_bus_yearly.rename(OUTCOME).reset_index()
        comp_y["treated"] = 0
        combined = pd.concat([subset_yearly, comp_y], ignore_index=True)
        combined = combined[~combined["year"].isin([2020, 2021])]
        combined = build_its_vars(combined)
        m = smf.ols(f"{OUTCOME} ~ (T + D1 + T1 + D2 + T2) * treated", data=combined).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        t2, t2p = m.params.get("T2:treated", float("nan")), m.pvalues.get("T2:treated", float("nan"))
        print(f"Dropping {dropped}: T2:treated={t2:+.2f} (p={t2p:.3f})")

   
    section(f"SECTION K2: Panel-level DiD on '{OUTCOME}' (clustered by count_point_id) vs A61")
    a647_panel_bus = a647_leeds[["count_point_id", "year", OUTCOME]].copy()
    a647_panel_bus["treated"] = 1
    comp_panel_bus = comp_src_bus[["count_point_id", "year", OUTCOME]].copy()
    comp_panel_bus["treated"] = 0
    panel_bus = pd.concat([a647_panel_bus, comp_panel_bus], ignore_index=True)
    panel_bus = panel_bus[~panel_bus["year"].isin([2020, 2021])]
    panel_bus = build_its_vars(panel_bus)

    ols_model_bus = smf.ols(f"{OUTCOME} ~ (T + D1 + T1 + D2 + T2) * treated", data=panel_bus)
    fit_bus = ols_model_bus.fit(cov_type="cluster", cov_kwds={"groups": panel_bus["count_point_id"]})

    print(f"\nPanel shape: {panel_bus.shape}, clusters: {panel_bus['count_point_id'].nunique()}")
    for term in ["D2:treated", "T2:treated"]:
        coef, p = fit_bus.params.get(term, float("nan")), fit_bus.pvalues.get(term, float("nan"))
        print(f"  {term}: {coef:+9.2f}  (p={p:.3f})")

    section(f"SECTION K3: Wild-cluster bootstrap on '{OUTCOME}' T2:treated (Rademacher, 9999 reps)")
    for param in ["D2:treated", "T2:treated"]:
        try:
            result = wildboottest(
                ols_model_bus,
                cluster=panel_bus["count_point_id"],
                param=param,
                bootstrap_type="11",
                B=9999,
            )
            print(f"\n{param}:")
            print(result)
        except KeyError:
            print(f"\n{param}: KeyError - check fit_bus.summary() for exact term name.")
            print(fit_bus.summary())

print("\nSection K complete — buses_and_coaches now has jackknife, cluster-robust, and bootstrap coverage.")



section("SECTION L: Structural break search, 2014-2022 (A647 alone)")

import numpy as np
from scipy import stats as sp_stats


def structural_break_search(series_df, outcome_col, candidate_years, exclude_covid=True):
    df_base = series_df.copy()
    if exclude_covid:
        df_base = df_base[~df_base["year"].isin([2020, 2021])]
    results = []
    for candidate in candidate_years:
        df = build_its_vars(df_base.copy(), lptip_year=candidate)
        full_model = smf.ols(f"{outcome_col} ~ T + D1 + T1 + D2 + T2", data=df).fit()
        restricted_model = smf.ols(f"{outcome_col} ~ T + D1 + T1", data=df).fit()
        rss_full = float(np.sum(full_model.resid ** 2))
        rss_restricted = float(np.sum(restricted_model.resid ** 2))
        df_full = full_model.df_resid
        df_restricted = restricted_model.df_resid
        q = df_restricted - df_full  
        if rss_full <= 0 or q <= 0:
            f_stat, p_value = float("nan"), float("nan")
        else:
            f_stat = ((rss_restricted - rss_full) / q) / (rss_full / df_full)
            p_value = 1 - sp_stats.f.cdf(f_stat, q, df_full)
        results.append({
            "candidate_year": candidate, "f_stat": round(f_stat, 3),
            "p_value": round(p_value, 4), "rss_full": round(rss_full, 1),
        })
    return pd.DataFrame(results).sort_values("f_stat", ascending=False).reset_index(drop=True)


candidate_years = list(range(2014, 2023))

for outcome in ["all_motor_vehicles", "buses_and_coaches"]:
    a647_series = a647_leeds.groupby("year")[outcome].mean().rename(outcome).reset_index()
    result_df = structural_break_search(a647_series, outcome, candidate_years, exclude_covid=True)
    print(f"\n--- Structural break search: {outcome} (A647 alone, excl. COVID) ---")
    print(result_df.to_string(index=False))
    best = result_df.iloc[0]
    print(f"Strongest additional break: {int(best['candidate_year'])} "
          f"(F={best['f_stat']}, p={best['p_value']})")



section("SECTION M: Synthetic control (A61 / A660 / A657 blend)")

from scipy.optimize import minimize

DONOR_ROADS = ["A61", "A660", "A657"]
PRE_PERIOD_CUTOFF = CS1_YEAR  


def build_yearly_series(road_name, outcome_col):
    return leeds[leeds["road_name"] == road_name].groupby("year")[outcome_col].mean()


def synthetic_control(outcome_col, exclude_covid=True):
    treated = a647_leeds.groupby("year")[outcome_col].mean()
    donors = {name: build_yearly_series(name, outcome_col) for name in DONOR_ROADS}

    all_years = treated.index
    if exclude_covid:
        all_years = [y for y in all_years if y not in (2020, 2021)]
    pre_years = [y for y in all_years if y < PRE_PERIOD_CUTOFF]
    post_lptip_years = [y for y in all_years if y >= CONSTRUCTION_START]

   
    common_pre_years = [y for y in pre_years if all(y in donors[d].index for d in DONOR_ROADS) and y in treated.index]
    if len(common_pre_years) < 3:
        print(f"  Insufficient common pre-period years for {outcome_col}: {common_pre_years}")
        return None

    treated_pre = treated.loc[common_pre_years].values
    donor_matrix = np.column_stack([donors[d].loc[common_pre_years].values for d in DONOR_ROADS])

    def objective(weights):
        synthetic = donor_matrix @ weights
        return np.sum((treated_pre - synthetic) ** 2)

    n_donors = len(DONOR_ROADS)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors
    init = np.ones(n_donors) / n_donors
    opt = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=constraints)

    weights = dict(zip(DONOR_ROADS, np.round(opt.x, 4)))
    pre_rmse = float(np.sqrt(opt.fun / len(common_pre_years)))
    pre_mean = float(np.mean(treated_pre))
    pre_rmse_pct = 100 * pre_rmse / pre_mean if pre_mean else float("nan")

    common_post_years = [y for y in post_lptip_years if all(y in donors[d].index for d in DONOR_ROADS) and y in treated.index]
    synthetic_post = sum(weights[d] * donors[d].loc[common_post_years].values for d in DONOR_ROADS)
    actual_post = treated.loc[common_post_years].values
    gap = actual_post - synthetic_post
    avg_gap = float(np.mean(gap)) if len(gap) else float("nan")

    print(f"\n--- Synthetic control: {outcome_col} ---")
    print(f"  Weights: {weights}")
    print(f"  Pre-2016 fit RMSE: {pre_rmse:.2f} ({pre_rmse_pct:.1f}% of pre-period mean)")
    print(f"  Post-2020 average gap (actual - synthetic): {avg_gap:+.2f}")
    print(f"  Post-2020 years used: {common_post_years}")
    return {"outcome": outcome_col, "weights": weights, "pre_rmse_pct": pre_rmse_pct, "avg_gap": avg_gap}


for outcome in ["all_motor_vehicles", "buses_and_coaches"]:
    synthetic_control(outcome, exclude_covid=True)

print("\nSections L-M complete (structural break search, synthetic control).")



section("SECTION N: Comparator treatment-status geographic check (A61, A660)")

A61_SOUTH_LANDMARKS = ["thwaite gate", "hunslet", "stourton", "low road", "hunslet road"]
A61_NORTH_LANDMARKS = ["sheepscar", "king lane", "stonegate", "grammar school", "scott hall", "potternewton"]
A660_LANDMARKS = ["lawnswood"]

def flag_landmarks(row, landmark_list):
    text = f"{row.get('start_junction_road_name', '')} {row.get('end_junction_road_name', '')}".lower()
    hits = [lm for lm in landmark_list if lm in text]
    return "; ".join(hits) if hits else ""

def geographic_check(road_name, landmark_sets):
    subset = leeds[leeds["road_name"] == road_name].drop_duplicates(subset="count_point_id").copy()
    subset = subset.sort_values("northing") if "northing" in subset.columns else subset.sort_values("latitude")
    print(f"\n--- {road_name}: {subset['count_point_id'].nunique()} count points, sorted south to north ---")
    for _, row in subset.iterrows():
        flags = []
        for label, landmarks in landmark_sets.items():
            hit = flag_landmarks(row, landmarks)
            if hit:
                flags.append(f"{label}[{hit}]")
        flag_str = " ".join(flags) if flags else "NO LANDMARK MATCH - manual check needed"
        print(f"  {row['count_point_id']:>7} | {str(row.get('start_junction_road_name','?')):30s} -> "
              f"{str(row.get('end_junction_road_name','?')):30s} | northing={row.get('northing','?')} | {flag_str}")

geographic_check("A61", {"A61_SOUTH": A61_SOUTH_LANDMARKS, "A61_NORTH": A61_NORTH_LANDMARKS})
geographic_check("A660", {"A660_LAWNSWOOD": A660_LANDMARKS})

print("\nNOTE: 'NO LANDMARK MATCH' does not confirm a point is untreated - it may sit on a")
print("continuously-treated stretch between two flagged points. Cross-check against official")
print("LPTIP scheme extent maps (e.g. West Yorkshire Combined Authority scheme location maps)")
print("before treating any point as a confirmed clean comparator.")


section("SECTION O: Counted vs Estimated data-quality audit")

AUDIT_ROADS = ["A647", "A61", "A660", "A64", "A65", "A657"]
JUMP_THRESHOLD = 0.30  

def data_quality_audit(road_name):
    subset = leeds[leeds["road_name"] == road_name].copy()
    print(f"\n--- {road_name}: data-quality summary by count point ---")
    summary_rows = []
    for cp_id, grp in subset.groupby("count_point_id"):
        grp = grp.sort_values("year")
        n_years = len(grp)
        method_col = "estimation_method" if "estimation_method" in grp.columns else None
        if method_col:
            counted = grp[method_col].astype(str).str.contains("Count", case=False, na=False).sum()
            estimated = n_years - counted
        else:
            counted, estimated = "?", "?"
        summary_rows.append({"count_point_id": cp_id, "n_years": n_years,
                              "n_counted": counted, "n_estimated": estimated})

       
        grp = grp.reset_index(drop=True)
        for i in range(1, len(grp)):
            prev, curr = grp.loc[i - 1], grp.loc[i]
            for outcome in ["all_motor_vehicles", "buses_and_coaches"]:
                if outcome not in grp.columns:
                    continue
                prev_val, curr_val = prev[outcome], curr[outcome]
                if prev_val and prev_val > 0:
                    pct_change = (curr_val - prev_val) / prev_val
                    method_changed = (method_col and prev[method_col] != curr[method_col])
                    if abs(pct_change) >= JUMP_THRESHOLD:
                        flag = " <-- COINCIDES WITH ESTIMATION-METHOD CHANGE" if method_changed else ""
                        print(f"  {cp_id} {outcome}: {int(prev['year'])}={prev_val:.0f} -> "
                              f"{int(curr['year'])}={curr_val:.0f} ({pct_change:+.0%}){flag}")
    print(f"\n  Coverage summary for {road_name}:")
    for r in summary_rows:
        print(f"    {r['count_point_id']}: {r['n_years']} years, "
              f"{r['n_counted']} counted, {r['n_estimated']} estimated")

for road in AUDIT_ROADS:
    data_quality_audit(road)

print("\nSections N-O complete (comparator geography check, data-quality audit).")



section("SECTION P: Time series data export for figures")

FIGURE_ROADS = ["A647", "A61", "A660"]

for road in FIGURE_ROADS:
    if road == "A647":
        subset = a647_leeds
    else:
        subset = leeds[leeds["road_name"] == road]
    for outcome in ["all_motor_vehicles", "buses_and_coaches"]:
        yearly = subset.groupby("year")[outcome].mean().round(1)
        print(f"\n--- {road} {outcome} (yearly mean across count points) ---")
        print(",".join(f"{int(y)}:{v}" for y, v in yearly.items()))

print("\n--- Count point coordinates (for map figure) ---")
for road in FIGURE_ROADS:
    if road == "A647":
        subset = a647_leeds
    else:
        subset = leeds[leeds["road_name"] == road]
    coords = subset.drop_duplicates(subset="count_point_id")[["count_point_id", "latitude", "longitude"]]
    print(f"\n{road}:")
    for _, row in coords.iterrows():
        print(f"  {row['count_point_id']}: lat={row['latitude']}, lon={row['longitude']}")

print("\nSection P complete (figure data export).")


section("SECTION Q: HAC lag sensitivity + full model reporting")

section("SECTION Q1: HAC lag sensitivity - buses_and_coaches, corridor-aggregate vs A61, excl. COVID")

a61_bus_yearly = leeds[leeds["road_name"] == "A61"].groupby("year")["buses_and_coaches"].mean()
a647_bus_yearly = a647_leeds.groupby("year")["buses_and_coaches"].mean()
a647_y = a647_bus_yearly.rename("buses_and_coaches").reset_index()
a647_y["treated"] = 1
comp_y = a61_bus_yearly.rename("buses_and_coaches").reset_index()
comp_y["treated"] = 0
combined_bus = pd.concat([a647_y, comp_y], ignore_index=True)
combined_bus = combined_bus[~combined_bus["year"].isin([2020, 2021])]
combined_bus = build_its_vars(combined_bus)

for lag in [1, 2, 3, 4]:
    m = smf.ols("buses_and_coaches ~ (T + D1 + T1 + D2 + T2) * treated", data=combined_bus).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag})
    coef = m.params["T2:treated"]
    se = m.bse["T2:treated"]
    ci_low, ci_high = m.conf_int().loc["T2:treated"]
    p = m.pvalues["T2:treated"]
    print(f"  maxlags={lag}: T2:treated={coef:+.2f} (SE={se:.2f}, 95% CI=[{ci_low:.2f}, {ci_high:.2f}], p={p:.4f}), n={int(m.nobs)}")

section("SECTION Q2: Full reporting for principal models (coef, SE, 95% CI, p, n, clusters)")

def report_full(model_fit, param, label, n_clusters=None):
    coef = model_fit.params[param]
    se = model_fit.bse[param]
    ci_low, ci_high = model_fit.conf_int().loc[param]
    p = model_fit.pvalues[param]
    n = int(model_fit.nobs)
    cluster_str = f", clusters={n_clusters}" if n_clusters else ""
    print(f"  {label}: {param}={coef:+.3f}, SE={se:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}], p={p:.4f}, n={n}{cluster_str}")

m_headline = smf.ols("buses_and_coaches ~ (T + D1 + T1 + D2 + T2) * treated", data=combined_bus).fit(
    cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
print("\n-- Principal model 1: Corridor-aggregate HAC, buses_and_coaches vs A61 --")
report_full(m_headline, "T2:treated", "Headline (excl. COVID)")
report_full(m_headline, "D2:treated", "Headline (excl. COVID)")

a647_panel_bus2 = a647_leeds[["count_point_id", "year", "buses_and_coaches"]].copy()
a647_panel_bus2["treated"] = 1
comp_panel_bus2 = leeds[leeds["road_name"] == "A61"][["count_point_id", "year", "buses_and_coaches"]].copy()
comp_panel_bus2["treated"] = 0
panel_bus2 = pd.concat([a647_panel_bus2, comp_panel_bus2], ignore_index=True)
panel_bus2 = panel_bus2[~panel_bus2["year"].isin([2020, 2021])]
panel_bus2 = build_its_vars(panel_bus2)
m_panel_bus = smf.ols("buses_and_coaches ~ (T + D1 + T1 + D2 + T2) * treated", data=panel_bus2).fit(
    cov_type="cluster", cov_kwds={"groups": panel_bus2["count_point_id"]})
print("\n-- Principal model 2: Panel cluster-robust, buses_and_coaches vs A61 --")
report_full(m_panel_bus, "T2:treated", "Panel cluster-robust", n_clusters=panel_bus2["count_point_id"].nunique())
report_full(m_panel_bus, "D2:treated", "Panel cluster-robust", n_clusters=panel_bus2["count_point_id"].nunique())

a647_panel_amv = a647_leeds[["count_point_id", "year", "all_motor_vehicles"]].copy()
a647_panel_amv["treated"] = 1
comp_panel_amv = leeds[leeds["road_name"] == "A61"][["count_point_id", "year", "all_motor_vehicles"]].copy()
comp_panel_amv["treated"] = 0
panel_amv = pd.concat([a647_panel_amv, comp_panel_amv], ignore_index=True)
panel_amv = panel_amv[~panel_amv["year"].isin([2020, 2021])]
panel_amv = build_its_vars(panel_amv)
m_panel_amv = smf.ols("all_motor_vehicles ~ (T + D1 + T1 + D2 + T2) * treated", data=panel_amv).fit(
    cov_type="cluster", cov_kwds={"groups": panel_amv["count_point_id"]})
print("\n-- Principal model 3: Panel cluster-robust, all_motor_vehicles vs A61 --")
report_full(m_panel_amv, "T2:treated", "Panel cluster-robust", n_clusters=panel_amv["count_point_id"].nunique())
report_full(m_panel_amv, "D2:treated", "Panel cluster-robust", n_clusters=panel_amv["count_point_id"].nunique())

print("\nSection Q complete (HAC lag sensitivity, full model reporting).")



section("SECTION R: Counted vs Estimated split by period (A647)")

PERIODS = {
    "2020-2022 (construction/COVID)": (2020, 2022),
    "2022-2025 (post-completion)": (2022, 2025),
}

method_col = "estimation_method" if "estimation_method" in a647_leeds.columns else None

for cp_id, grp in a647_leeds.groupby("count_point_id"):
    print(f"\n--- Count point {cp_id} ---")
    for label, (start, end) in PERIODS.items():
        period_grp = grp[(grp["year"] >= start) & (grp["year"] <= end)].sort_values("year")
        if method_col:
            counted = period_grp[method_col].astype(str).str.contains("Count", case=False, na=False)
            n_counted = counted.sum()
            n_total = len(period_grp)
            years_str = ", ".join(
                f"{int(r['year'])}={'C' if c else 'E'}"
                for (_, r), c in zip(period_grp.iterrows(), counted)
            )
            print(f"  {label}: {n_counted}/{n_total} counted -> [{years_str}]")
        else:
            print(f"  {label}: estimation_method field not found")

print("\nSection R complete (Counted vs Estimated split by period).")



section("SECTION S: Link-definition check for flagged A647 discontinuities")

LINK_FIELDS = [c for c in ["start_junction_road_name", "end_junction_road_name", "link_length_km"]
               if c in a647_leeds.columns]

if not LINK_FIELDS:
    print("None of start_junction_road_name / end_junction_road_name / link_length_km found in the data - cannot run this check.")
else:
    print(f"Checking fields: {LINK_FIELDS}\n")
    for cp_id, grp in a647_leeds.groupby("count_point_id"):
        grp = grp.sort_values("year")
        print(f"--- Count point {cp_id} ---")
        prev_vals = None
        for _, row in grp.iterrows():
            vals = tuple(row[f] for f in LINK_FIELDS)
            changed = " <-- CHANGED FROM PRIOR YEAR" if prev_vals is not None and vals != prev_vals else ""
            if changed or prev_vals is None:
                vals_str = ", ".join(f"{f}={row[f]}" for f in LINK_FIELDS)
                print(f"  {int(row['year'])}: {vals_str}{changed}")
            prev_vals = vals
        print()

print("Section S complete (link-definition check).")



section("SECTION T: Estimation-artefact vs genuine-change check")

section("SECTION T1: Full per-point trajectory around flagged jumps")

FLAGGED_POINTS = {
    28289: ("all_motor_vehicles", 2016),
    27428: ("all_motor_vehicles", 2017),
}

for cp_id, (outcome, jump_year) in FLAGGED_POINTS.items():
    grp = a647_leeds[a647_leeds["count_point_id"] == cp_id].sort_values("year")
    method_col = "estimation_method" if "estimation_method" in grp.columns else None
    print(f"\n--- Count point {cp_id}, {outcome}, jump year {jump_year} ---")
    for _, row in grp.iterrows():
        marker = " <-- JUMP YEAR" if int(row["year"]) == jump_year else ""
        method = f", method={row[method_col]}" if method_col else ""
        print(f"  {int(row['year'])}: {row[outcome]:.0f}{method}{marker}")

section("SECTION T2: Network-wide check - how many Leeds count points show a similar jump in the same year")

for cp_id, (outcome, jump_year) in FLAGGED_POINTS.items():
    prev_year, this_year = jump_year - 1, jump_year
    yearly = leeds[leeds["year"].isin([prev_year, this_year])][["count_point_id", "road_name", "year", outcome]]
    pivot = yearly.pivot_table(index=["count_point_id", "road_name"], columns="year", values=outcome)
    pivot = pivot.dropna()
    if prev_year in pivot.columns and this_year in pivot.columns:
        pivot["pct_change"] = (pivot[this_year] - pivot[prev_year]) / pivot[prev_year] * 100
        big_jumps = pivot[pivot["pct_change"].abs() >= 30]
        print(f"\n--- {outcome}, {prev_year}->{this_year}: count points network-wide with |change| >= 30% ---")
        print(f"Total count points compared: {len(pivot)}; number with a jump >= 30%: {len(big_jumps)} "
              f"({100*len(big_jumps)/len(pivot):.1f}% of all compared points)")
        print(big_jumps.sort_values("pct_change", ascending=False).head(15))

print("\nSection T complete (estimation-artefact vs genuine-change check).")



section("SECTION U: Systematic Estimated<->Counted transition check, all A647 points")

method_col = "estimation_method" if "estimation_method" in a647_leeds.columns else None

if not method_col:
    print("estimation_method field not found - cannot run this check.")
else:
    transitions = []
    for cp_id, grp in a647_leeds.groupby("count_point_id"):
        grp = grp.sort_values("year").reset_index(drop=True)
        for i in range(1, len(grp)):
            prev, curr = grp.loc[i - 1], grp.loc[i]
            prev_method = "Counted" if "Count" in str(prev[method_col]) else "Estimated"
            curr_method = "Counted" if "Count" in str(curr[method_col]) else "Estimated"
            if prev_method != curr_method:
                for outcome in ["all_motor_vehicles", "buses_and_coaches"]:
                    if outcome not in grp.columns or not prev[outcome] or prev[outcome] == 0:
                        continue
                    pct_change = (curr[outcome] - prev[outcome]) / prev[outcome] * 100
                    transitions.append({
                        "count_point_id": cp_id, "year": int(curr["year"]), "outcome": outcome,
                        "transition": f"{prev_method}->{curr_method}", "pct_change": pct_change,
                    })

    trans_df = pd.DataFrame(transitions)
    print(f"Total Estimated<->Counted transitions found across all 5 A647 points, both outcomes: {len(trans_df)}\n")

    for direction in ["Estimated->Counted", "Counted->Estimated"]:
        for outcome in ["all_motor_vehicles", "buses_and_coaches"]:
            subset = trans_df[(trans_df["transition"] == direction) & (trans_df["outcome"] == outcome)]
            if len(subset) == 0:
                continue
            print(f"--- {direction}, {outcome}: n={len(subset)} ---")
            print(f"  Mean % change: {subset['pct_change'].mean():+.1f}%")
            print(f"  Median % change: {subset['pct_change'].median():+.1f}%")
            print(f"  Std dev: {subset['pct_change'].std():.1f}")
            print(f"  Min / Max: {subset['pct_change'].min():+.1f}% / {subset['pct_change'].max():+.1f}%")
            n_positive = (subset["pct_change"] > 0).sum()
            print(f"  Positive (jump up): {n_positive}/{len(subset)} ({100*n_positive/len(subset):.0f}%)")
            print()

    print("Full transition list:")
    print(trans_df.sort_values(["outcome", "transition", "pct_change"]).to_string(index=False))

print("\nSection U complete (systematic Estimated<->Counted transition check).")


section("SECTION V: Systematic Estimated<->Counted transition check, A61 and A660")

BREAKPOINT_YEARS = {2016, 2020}
method_col = "estimation_method" if "estimation_method" in leeds.columns else None

if not method_col:
    print("estimation_method field not found - cannot run this check.")
else:
    for comparator_road in ["A61", "A660"]:
        comp_data = leeds[leeds["road_name"] == comparator_road]
        transitions = []
        for cp_id, grp in comp_data.groupby("count_point_id"):
            grp = grp.sort_values("year").reset_index(drop=True)
            for i in range(1, len(grp)):
                prev, curr = grp.loc[i - 1], grp.loc[i]
                prev_method = "Counted" if "Count" in str(prev[method_col]) else "Estimated"
                curr_method = "Counted" if "Count" in str(curr[method_col]) else "Estimated"
                if prev_method != curr_method:
                    for outcome in ["all_motor_vehicles", "buses_and_coaches"]:
                        if outcome not in grp.columns or not prev[outcome] or prev[outcome] == 0:
                            continue
                        pct_change = (curr[outcome] - prev[outcome]) / prev[outcome] * 100
                        transitions.append({
                            "count_point_id": cp_id, "year": int(curr["year"]), "outcome": outcome,
                            "transition": f"{prev_method}->{curr_method}", "pct_change": pct_change,
                            "on_breakpoint_year": int(curr["year"]) in BREAKPOINT_YEARS,
                        })

        trans_df = pd.DataFrame(transitions)
        print(f"\n=== {comparator_road}: {len(trans_df)} total Estimated<->Counted transitions found ===")

        for direction in ["Estimated->Counted", "Counted->Estimated"]:
            for outcome in ["all_motor_vehicles", "buses_and_coaches"]:
                subset = trans_df[(trans_df["transition"] == direction) & (trans_df["outcome"] == outcome)]
                if len(subset) == 0:
                    continue
                n_positive = (subset["pct_change"] > 0).sum()
                print(f"  {direction}, {outcome}: n={len(subset)}, mean={subset['pct_change'].mean():+.1f}%, "
                      f"median={subset['pct_change'].median():+.1f}%, positive={n_positive}/{len(subset)}")

        on_bp = trans_df[(trans_df["on_breakpoint_year"]) & (trans_df["pct_change"].abs() >= 20)]
        print(f"\n  Transitions landing on a breakpoint year (2016 or 2020) with |change| >= 20%: {len(on_bp)}")
        if len(on_bp) > 0:
            print(on_bp[["count_point_id", "year", "outcome", "transition", "pct_change"]].to_string(index=False))

print("\nSection V complete (comparator-side systematic transition check).")



section("SECTION W: A660 (excl. count point 57490) as comparator - full rerun")

a660_clean = leeds[(leeds["road_name"] == "A660") & (leeds["count_point_id"] != 57490)]
print(f"A660 (excl. 57490): {a660_clean['count_point_id'].nunique()} count points")

section("SECTION W1: Corridor-aggregate DiD vs A660-clean, both outcomes, full and excl. COVID")

for outcome in ["all_motor_vehicles", "buses_and_coaches"]:
    a647_y = a647_leeds.groupby("year")[outcome].mean().rename(outcome).reset_index()
    a647_y["treated"] = 1
    comp_y = a660_clean.groupby("year")[outcome].mean().rename(outcome).reset_index()
    comp_y["treated"] = 0
    combined = pd.concat([a647_y, comp_y], ignore_index=True)

    for spec_label, spec_df in [("full", combined), ("excl_covid", combined[~combined["year"].isin([2020, 2021])])]:
        spec_df = build_its_vars(spec_df)
        m = smf.ols(f"{outcome} ~ (T + D1 + T1 + D2 + T2) * treated", data=spec_df).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        print(f"  {outcome} [{spec_label:11s}] vs A660-clean: "
              f"D2:treated={m.params['D2:treated']:+9.1f} (p={m.pvalues['D2:treated']:.3f})   "
              f"T2:treated={m.params['T2:treated']:+9.1f} (p={m.pvalues['T2:treated']:.3f})")

section("SECTION W2: Jackknife vs A660-clean, buses_and_coaches, excl. COVID")

a660_bus_yearly = a660_clean.groupby("year")["buses_and_coaches"].mean()
for drop_id in a647_leeds["count_point_id"].unique():
    a647_sub = a647_leeds[a647_leeds["count_point_id"] != drop_id]
    a647_y = a647_sub.groupby("year")["buses_and_coaches"].mean().rename("buses_and_coaches").reset_index()
    a647_y["treated"] = 1
    comp_y = a660_bus_yearly.rename("buses_and_coaches").reset_index()
    comp_y["treated"] = 0
    combined = pd.concat([a647_y, comp_y], ignore_index=True)
    combined = combined[~combined["year"].isin([2020, 2021])]
    combined = build_its_vars(combined)
    m = smf.ols("buses_and_coaches ~ (T + D1 + T1 + D2 + T2) * treated", data=combined).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    print(f"  Dropping {drop_id}: T2:treated={m.params['T2:treated']:+.2f} (p={m.pvalues['T2:treated']:.3f})")

section("SECTION W3: Panel cluster-robust DiD vs A660-clean, buses_and_coaches, excl. COVID")

a647_panel = a647_leeds[["count_point_id", "year", "buses_and_coaches"]].copy()
a647_panel["treated"] = 1
comp_panel = a660_clean[["count_point_id", "year", "buses_and_coaches"]].copy()
comp_panel["treated"] = 0
panel = pd.concat([a647_panel, comp_panel], ignore_index=True)
panel = panel[~panel["year"].isin([2020, 2021])]
panel = build_its_vars(panel)
n_clusters = panel["count_point_id"].nunique()


ols_model = smf.ols("buses_and_coaches ~ (T + D1 + T1 + D2 + T2) * treated", data=panel)
m_panel = ols_model.fit(cov_type="cluster", cov_kwds={"groups": panel["count_point_id"]})
print(f"  Panel shape: {panel.shape}, clusters: {n_clusters}")
print(f"  D2:treated: {m_panel.params['D2:treated']:+9.2f} (SE={m_panel.bse['D2:treated']:.2f}, p={m_panel.pvalues['D2:treated']:.3f})")
print(f"  T2:treated: {m_panel.params['T2:treated']:+9.2f} (SE={m_panel.bse['T2:treated']:.2f}, p={m_panel.pvalues['T2:treated']:.3f})")

print("\n  Wild-cluster bootstrap (Rademacher weights, 9999 reps) - note only 14 clusters, well below")
print("  the conventional 30-50 threshold, so this check matters more here than for the A61 panel:")
for param in ["D2:treated", "T2:treated"]:
    try:
        result = wildboottest(
            ols_model,
            cluster=panel["count_point_id"],
            param=param,
            bootstrap_type="11",
            B=9999,
        )
        print(f"\n{param}:")
        print(result)
    except KeyError:
        print(f"\n{param}: KeyError - check m_panel.summary() below for the exact term name.")
        print(m_panel.summary())
    except Exception as e:
        print(f"\n{param}: bootstrap failed with: {e}")

print("\nSection W complete (A660-clean comparator full rerun).")


section("SECTION X: Bus/coach-specific pre-trend test")

a647_leeds_yearly_bus = a647_leeds.groupby("year")["buses_and_coaches"].mean()

screening_bus = []

for road in CANDIDATE_LEEDS_ROADS:
    comp = leeds[leeds["road_name"] == road]
    n_points = comp["count_point_id"].nunique()
    comp_yearly = comp.groupby("year")["buses_and_coaches"].mean()
    r = pretrend_test(a647_leeds_yearly_bus, comp_yearly, road)
    r["n_count_points"] = n_points
    screening_bus.append(r)

single = bradford[bradford["count_point_id"] == BRADFORD_BOUNDARY_POINT]
single_yearly = single.groupby("year")["buses_and_coaches"].mean()
r = pretrend_test(a647_leeds_yearly_bus, single_yearly, "Bradford single point (73114)")
r["n_count_points"] = 1
screening_bus.append(r)

coverage = a647_bradford_all.groupby("count_point_id")["year"].nunique()
well_covered = coverage[coverage >= PRETREND_MIN_YEARS].index.tolist()
stretch = a647_bradford_all[a647_bradford_all["count_point_id"].isin(well_covered)]
stretch_yearly = stretch.groupby("year")["buses_and_coaches"].mean()
r = pretrend_test(a647_leeds_yearly_bus, stretch_yearly, "Bradford whole stretch (avg)")
r["n_count_points"] = len(well_covered)
screening_bus.append(r)

screening_bus_df = pd.DataFrame(screening_bus)[["comparator", "n_count_points", "n_obs", "slope_diff", "p_value", "result"]]
print(screening_bus_df.to_string(index=False))
screening_bus_df.to_csv("comparator_screening_buses_v1.csv", index=False)
print("\nSaved comparator_screening_buses_v1.csv")

passing_bus = screening_bus_df[screening_bus_df["result"] == "PASS (parallel)"].sort_values("n_count_points", ascending=False)
print(f"\nComparators that PASS the bus/coach-specific pre-trend test: {passing_bus['comparator'].tolist()}")

print("\nComparison with the all_motor_vehicles screening (Section B):")
print("  A61  : all_motor_vehicles p=0.632 (PASS)  |  buses_and_coaches p=? (see above)")
print("  A660 : all_motor_vehicles p=0.112 (PASS)  |  buses_and_coaches p=? (see above)")

print("\nSection X complete (bus/coach-specific pre-trend test).")



section("SECTION Y: 2016-2019 comparator trajectory check (buses/coaches)")

a647_bus_yearly_all = a647_leeds.groupby("year")["buses_and_coaches"].mean()

for comparator_road in ["A61", "A660"]:
    comp_yearly = leeds[leeds["road_name"] == comparator_road].groupby("year")["buses_and_coaches"].mean()
    print(f"\n--- A647 vs {comparator_road}, buses/coaches, 2016-2019 ---")
    for year in [2016, 2017, 2018, 2019]:
        a647_val = a647_bus_yearly_all.get(year, float("nan"))
        comp_val = comp_yearly.get(year, float("nan"))
        print(f"  {year}: A647={a647_val:.1f}, {comparator_road}={comp_val:.1f}")

    a647_start, a647_end = a647_bus_yearly_all[2016], a647_bus_yearly_all[2019]
    comp_start, comp_end = comp_yearly[2016], comp_yearly[2019]
    a647_pct = (a647_end - a647_start) / a647_start * 100
    comp_pct = (comp_end - comp_start) / comp_start * 100
    print(f"  Net change 2016-2019: A647={a647_pct:+.1f}%, {comparator_road}={comp_pct:+.1f}%")

    window = pd.concat([
        a647_bus_yearly_all[a647_bus_yearly_all.index.isin([2016, 2017, 2018, 2019])].rename("value").reset_index().assign(treated=1),
        comp_yearly[comp_yearly.index.isin([2016, 2017, 2018, 2019])].rename("value").reset_index().assign(treated=0),
    ], ignore_index=True)
    window["year_c"] = window["year"] - 2016
    m = smf.ols("value ~ year_c * treated", data=window).fit()
    coef = m.params.get("year_c:treated", float("nan"))
    pval = m.pvalues.get("year_c:treated", float("nan"))
    print(f"  2016-2019 slope-difference test: coef={coef:.2f}, p={pval:.3f} "
          f"({'PASS (similar 2016-19 trend)' if pval > 0.05 else 'FAIL (differs)'})")

print("\nSection Y complete (2016-2019 comparator trajectory check).")
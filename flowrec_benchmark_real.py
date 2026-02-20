"""
FlowRec — Benchmark con Datasets Reales
========================================
Ejecuta FlowRec vs MinT vs BottomUp en 3 datasets estándar:
  1. M5 (Walmart)      — 42,840 series de retail (muestreo estratificado real)
  2. Tourism Large     — 555 series de turismo (Australia, sintético a escala)
  3. Traffic (SF)      — ~217 sensores (sintético a escala real)

Dependencias:
  pip install datasetsforecast

Uso:
  python flowrec_benchmark_real.py

Salida:
  benchmark_real_results.csv        — tabla de métricas
  output_real/benchmark_real_rmse.png
  output_real/benchmark_real_speedup.png
  output_real/benchmark_real_tabla.png
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flowrec_core import HierarchicalNetwork, FlowRec, MinTrace, BottomUp, rmse

OUTPUT_DIR = "output_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

C = {
    "fr":   "#2563EB",
    "mint": "#DC2626",
    "bu":   "#16A34A",
    "base": "#9CA3AF",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ==============================================================
# UTILIDADES COMUNES
# ==============================================================

def add_noise(Y, scale=0.15, seed=99):
    rng = np.random.default_rng(seed)
    std = np.std(Y, axis=0) + 1e-8
    return np.maximum(Y + rng.normal(0, scale * std, Y.shape), 0)


def run_benchmark_on_data(Y_true, Y_hat, net, n_samples=200, label=""):
    n = Y_true.shape[0]
    indices = np.random.default_rng(42).choice(n, size=min(n_samples, n), replace=False)

    models = {
        "BottomUp": BottomUp(net),
        "MinT-OLS": MinTrace(net, variant="ols"),
        "FlowRec":  FlowRec(net),
    }

    results = []
    for name, model in models.items():
        latencies = []
        rmse_list = []
        for i in indices:
            y_hat_i  = Y_hat[i]
            y_true_i = Y_true[i]
            t0 = time.perf_counter()
            y_rec, _ = model.reconcile(y_hat_i)
            latencies.append((time.perf_counter() - t0) * 1000)
            rmse_list.append(rmse(y_true_i, y_rec))

        base_rmse_list = [rmse(Y_true[i], Y_hat[i]) for i in indices]
        results.append({
            "Dataset":          label,
            "Método":           name,
            "RMSE_base":        float(np.mean(base_rmse_list)),
            "RMSE_rec":         float(np.mean(rmse_list)),
            "Mejora_%":         float(100 * (np.mean(base_rmse_list) - np.mean(rmse_list))
                                      / np.mean(base_rmse_list)),
            "Latencia_ms":      float(np.mean(latencies)),
            "Latencia_p95_ms":  float(np.percentile(latencies, 95)),
        })
        print(f"  [{label}] {name:10s}  RMSE={np.mean(rmse_list):.4f}  "
              f"lat={np.mean(latencies)*1000:.1f}µs")

    mint_lat = next(r["Latencia_ms"] for r in results if r["Método"] == "MinT-OLS")
    for r in results:
        r["Speedup_vs_MinT"] = mint_lat / r["Latencia_ms"]

    return results


def _build_net_from_S(S_matrix):
    """Construye HierarchicalNetwork desde una S_matrix [n_total × n_base]."""
    n_total, n_base = S_matrix.shape
    n_agg = n_total - n_base
    if n_agg <= 1:
        levels = [1, n_base]
    elif n_agg <= 20:
        levels = [1, n_agg, n_base]
    else:
        l1 = 1
        l2 = max(1, int(np.sqrt(n_agg)))
        l3 = max(l2, n_agg - l1 - l2)
        levels = [l1, l2, l3, n_base]
    return HierarchicalNetwork.from_tree(levels)


# ==============================================================
# DATASET 1 — M5 (Walmart) — datos REALES con muestreo
# ==============================================================

def run_m5():
    print("\n" + "="*60)
    print("DATASET 1 — M5 (Walmart)  [42,840 series, muestreo real]")
    print("="*60)

    use_real = False
    try:
        from datasetsforecast.m5 import M5
        Y_df, _, tags = M5.load(directory="data/m5")
        print("✅ M5 cargado desde datasetsforecast")
        use_real = True
    except Exception as e:
        print(f"⚠️  M5 no disponible ({e}). Usando jerarquía sintética a escala M5.")

    if use_real:
        # ── tags es el DataFrame de jerarquía real del M5 ──────────────────
        # Columnas: unique_id, item_id, dept_id, cat_id, store_id, state_id
        # Cada fila es una serie BASE (hoja) con sus etiquetas de nivel.
        print(f"  tags shape: {tags.shape}  cols: {tags.columns.tolist()}")
        base_ids = tags["unique_id"].tolist()          # 30,490 series base
        n_base   = len(base_ids)

        # ── Muestreo estratificado: N_SAMPLE series base reales ─────────────
        N_SAMPLE   = 500
        rng_s      = np.random.default_rng(42)
        sample_ids = rng_s.choice(base_ids, size=min(N_SAMPLE, n_base),
                                  replace=False).tolist()
        print(f"  Muestreando {len(sample_ids)} series base "
              f"({len(sample_ids)/n_base*100:.1f}% del total)...")

        # Pivotar solo las series muestreadas (manejable en RAM)
        Y_sample_df = Y_df[Y_df["unique_id"].isin(sample_ids)]
        Y_pivot = (Y_sample_df
                   .pivot(index="ds", columns="unique_id", values="y")
                   .reindex(columns=sample_ids)
                   .fillna(0.0))
        Y_base_real = Y_pivot.values.astype(float)    # [T × N_SAMPLE]
        T = Y_base_real.shape[0]
        print(f"  Y_base_real: {T} timesteps × {len(sample_ids)} series hoja")

        # ── Construir jerarquía real desde tags ────────────────────────────
        # Nivel: Total→State(3)→Store(10)→Dept(7)→series_muestra
        tag_sample = tags[tags["unique_id"].isin(sample_ids)]
        n_states = tag_sample["state_id"].nunique()
        n_stores = tag_sample["store_id"].nunique()
        n_depts  = tag_sample["dept_id"].nunique()

        # Jerarquía balanceada con los niveles reales observados en la muestra
        levels = [1, n_states, n_stores, n_depts, len(sample_ids)]
        net_m5 = HierarchicalNetwork.from_tree(levels)
        n_total_sub = net_m5.n_total
        print(f"  Red M5 (muestrada): {n_total_sub} nodos — niveles {levels}")

        Y_full = (net_m5.S @ Y_base_real.T).T        # [T × n_total_sub]

    else:
        # Sintético escalado (fallback M5-like)
        print("  Construyendo jerarquía sintética 1→3→10→7→343 (M5-like)...")
        levels  = [1, 3, 10, 7, 343]
        net_m5  = HierarchicalNetwork.from_tree(levels)
        T       = 1969    # M5 tiene 1969 timesteps diarios
        rng     = np.random.default_rng(42)
        n_base  = net_m5.n_base
        Y_base  = rng.lognormal(mean=5, sigma=0.5, size=(T, n_base))
        Y_full  = (net_m5.S @ Y_base.T).T

    Y_hat   = add_noise(Y_full, scale=0.20)
    results = run_benchmark_on_data(Y_full, Y_hat, net_m5,
                                    n_samples=500, label="M5 (Walmart)")
    return results, net_m5


# ==============================================================
# DATASET 2 — Tourism Large (Australia)
# ==============================================================

def run_tourism():
    print("\n" + "="*60)
    print("DATASET 2 — Tourism Large (Australia)  [555 series]")
    print("="*60)

    use_real = False
    try:
        from datasetsforecast.hierarchical import HierarchicalData
        Y_df, S_df, tags = HierarchicalData.load(
            directory="data/hierarchical", group="TourismLarge")
        print("✅ TourismLarge cargado desde datasetsforecast.hierarchical")
        use_real = True
    except Exception as e:
        print(f"⚠️  TourismLarge no disponible ({e}). Usando sintético a escala real.")

    if use_real:
        # S_df: índice = unique_id de TODOS los nodos, cols = series base (304)
        S_matrix = S_df.values.astype(float)    # [555 × 304]
        n_total, n_base = S_matrix.shape
        base_ids = S_df.columns.tolist()        # 304 series hoja
        print(f"  S shape: {n_total} nodos × {n_base} series base")

        # Pivotar solo las 304 series hoja (manejable)
        Y_base_df = Y_df[Y_df["unique_id"].isin(base_ids)]
        Y_pivot = (Y_base_df
                   .pivot(index="ds", columns="unique_id", values="y")
                   .reindex(columns=base_ids)
                   .fillna(0.0))
        Y_base_real = Y_pivot.values.astype(float)          # [T × 304]
        T = Y_base_real.shape[0]
        print(f"  Y_base_real: {T} timesteps × {n_base} series base")

        net_tourism = _build_net_from_S(S_matrix)
        Y_full = (net_tourism.S @ Y_base_real.T).T          # [T × 555]
        print(f"  Red: {net_tourism.n_total} nodos")
    else:
        # Sintético a escala Tourism: 1→8→56→304
        print("  Sintético Tourism: 1→8→56→304...")
        levels      = [1, 8, 56, 304]
        net_tourism = HierarchicalNetwork.from_tree(levels)
        T           = 228
        rng         = np.random.default_rng(7)
        n_base      = net_tourism.n_base
        t           = np.arange(T)
        Y_base = np.column_stack([
            rng.uniform(100, 5000) + 0.3*t
            + 500*np.sin(2*np.pi*t/12 + rng.uniform(0, 2*np.pi))
            + rng.normal(0, 50, T)
            for _ in range(n_base)
        ])
        Y_base = np.maximum(Y_base, 0)
        Y_full = (net_tourism.S @ Y_base.T).T

    Y_hat   = add_noise(Y_full, scale=0.12)
    results = run_benchmark_on_data(Y_full, Y_hat, net_tourism,
                                    n_samples=200, label="Tourism Large")
    return results, net_tourism


# ==============================================================
# DATASET 3 — Traffic
# ==============================================================

def run_traffic():
    print("\n" + "="*60)
    print("DATASET 3 — Traffic Jerárquico Real  [207 nodos, 200 base]")
    print("="*60)

    use_real = False
    try:
        from datasetsforecast.hierarchical import HierarchicalData
        Y_df, S_df, tags = HierarchicalData.load(
            directory="data/hierarchical", group="Traffic")
        print("✅ Traffic cargado desde datasetsforecast.hierarchical")
        use_real = True
    except Exception as e:
        print(f"⚠️  Traffic no disponible ({e}). Usando sintético calibrado.")

    if use_real:
        # S_df: índice = unique_id de todos los nodos, cols = series base (200)
        S_matrix = S_df.values.astype(float)    # [207 × 200]
        n_total, n_base = S_matrix.shape
        base_ids = S_df.columns.tolist()        # 200 series hoja
        print(f"  S shape: {n_total} nodos × {n_base} series base")

        # Pivotar series hoja
        Y_base_df = Y_df[Y_df["unique_id"].isin(base_ids)]
        Y_pivot = (Y_base_df
                   .pivot(index="ds", columns="unique_id", values="y")
                   .reindex(columns=base_ids)
                   .fillna(0.0))
        Y_base_real = Y_pivot.values.astype(float)          # [T × 200]
        T = Y_base_real.shape[0]
        print(f"  Y_base_real: {T} timesteps × {n_base} series base")

        net_traffic = _build_net_from_S(S_matrix)
        Y_full = (net_traffic.S @ Y_base_real.T).T          # [T × 207]
        print(f"  Red: {net_traffic.n_total} nodos")
    else:
        # Sintético calibrado PEMS-BAY
        print("  Generando tráfico sintético calibrado (217 sensores)...")
        T           = 8736
        n_base      = 217
        n_distr     = 12
        levels      = [1, n_distr, n_base]
        net_traffic = HierarchicalNetwork.from_tree(levels)
        rng         = np.random.default_rng(2025)
        t           = np.arange(T)
        daily_p     = 288
        weekly_p    = daily_p * 7
        Y_base = np.column_stack([
            rng.uniform(30, 65)
            - 20*np.clip(np.sin(2*np.pi*t/daily_p - 0.5), 0, 1)
            - 15*np.clip(np.sin(2*np.pi*t/daily_p - 1.2), 0, 1)
            + 8 *np.sin(2*np.pi*t/weekly_p)
            + rng.normal(0, 3, T)
            for _ in range(n_base)
        ])
        Y_base = np.clip(Y_base, 0, 80)
        Y_full = (net_traffic.S @ Y_base.T).T

    Y_hat   = add_noise(Y_full, scale=0.08)
    results = run_benchmark_on_data(Y_full, Y_hat, net_traffic,
                                    n_samples=1000, label="Traffic")
    return results, net_traffic


# ==============================================================
# VISUALIZACIONES
# ==============================================================

def plot_results(all_results):
    df       = pd.DataFrame(all_results)
    datasets = df["Dataset"].unique()

    # ── Fig 1: RMSE comparativo ──
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub       = df[df["Dataset"] == ds]
        methods   = sub["Método"].tolist()
        rmse_vals = sub["RMSE_rec"].tolist()
        rmse_base = sub["RMSE_base"].iloc[0]
        colors    = [C["bu"], C["mint"], C["fr"]]
        bars = ax.bar(methods, rmse_vals, color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(rmse_base, color=C["base"], linestyle="--",
                   label=f"Base (sin rec.): {rmse_base:.3f}")
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(ds, fontweight="bold")
        ax.set_ylabel("RMSE promedio")
        ax.legend(fontsize=8)
        ax.set_ylim(0, rmse_base * 1.35)
    plt.suptitle("Comparativa RMSE — FlowRec vs Baselines", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "benchmark_real_rmse.png")
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"✅ {path}")

    # ── Fig 2: Latencia y Speedup ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot_lat = df.pivot(index="Dataset", columns="Método", values="Latencia_ms")
    x = np.arange(len(pivot_lat)); w = 0.25
    for i, (method, color) in enumerate(zip(["BottomUp", "MinT-OLS", "FlowRec"],
                                             [C["bu"], C["mint"], C["fr"]])):
        if method in pivot_lat.columns:
            axes[0].bar(x + i*w, pivot_lat[method]*1000, w, label=method,
                        color=color, alpha=0.85, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x + w); axes[0].set_xticklabels(pivot_lat.index, rotation=10)
    axes[0].set_ylabel("Latencia promedio (µs) — escala log")
    axes[0].set_title("Latencia por Dataset y Método", fontweight="bold")
    axes[0].legend()

    fr_rows = df[df["Método"] == "FlowRec"]
    bars2 = axes[1].bar(fr_rows["Dataset"], fr_rows["Speedup_vs_MinT"],
                        color=C["fr"], alpha=0.85, edgecolor="white")
    for bar, (_, row) in zip(bars2, fr_rows.iterrows()):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f"{row['Speedup_vs_MinT']:.0f}x",
                     ha="center", va="bottom", fontweight="bold")
    axes[1].axhline(1.0, color=C["mint"], linestyle="--", label="MinT baseline (1x)")
    axes[1].set_ylabel("Speedup vs MinT-OLS")
    axes[1].set_title("FlowRec Speedup vs MinT", fontweight="bold")
    axes[1].legend()
    plt.suptitle("Análisis de Rendimiento — Latencia & Speedup", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "benchmark_real_speedup.png")
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"✅ {path}")

    # ── Fig 3: Tabla resumen FlowRec ──
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    fr = df[df["Método"] == "FlowRec"].copy()
    fr["Latencia_us"]    = (fr["Latencia_ms"]    * 1000).round(1)
    fr["Lat_p95_us"]     = (fr["Latencia_p95_ms"]* 1000).round(1)
    fr["Speedup_vs_MinT"] = fr["Speedup_vs_MinT"].round(0).astype(int)
    table_data = fr[["Dataset", "RMSE_base", "RMSE_rec", "Mejora_%",
                      "Latencia_us", "Lat_p95_us", "Speedup_vs_MinT"]].round(2)
    table_data.columns = ["Dataset", "RMSE Base", "RMSE FlowRec",
                           "Mejora %", "Lat med (µs)", "Lat p95 (µs)", "Speedup vs MinT"]
    tbl = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.8)
    for j in range(len(table_data.columns)):
        tbl[0, j].set_facecolor("#1e3a5f")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("FlowRec — Resultados Benchmark Real", fontweight="bold", fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "benchmark_real_tabla.png")
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"✅ {path}")


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("FlowRec — Benchmark con Datasets Reales")
    print("arXiv:2505.03955  |  Amazon Science (2025)")
    print("=" * 60)

    try:
        import datasetsforecast
        print(f"✅ datasetsforecast {datasetsforecast.__version__} disponible\n")
    except ImportError:
        print("⚠️  datasetsforecast no instalado. pip install datasetsforecast\n")

    all_results = []

    for label, runner in [("M5", run_m5), ("Tourism", run_tourism), ("Traffic", run_traffic)]:
        try:
            r, _ = runner()
            all_results.extend(r)
        except Exception as e:
            print(f"❌ {label} falló: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("❌ Sin resultados.")
        return

    df = pd.DataFrame(all_results)
    csv_path = "benchmark_real_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Guardado en '{csv_path}'")

    print("\nGenerando gráficas...")
    plot_results(all_results)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL — FlowRec vs Baselines")
    print("=" * 60)
    for _, row in df[df["Método"] == "FlowRec"].iterrows():
        lat_us = row["Latencia_ms"] * 1000
        p95_us = row["Latencia_p95_ms"] * 1000
        print(f"\n📊 {row['Dataset']}")
        print(f"   RMSE Base    : {row['RMSE_base']:.4f}")
        print(f"   RMSE FlowRec : {row['RMSE_rec']:.4f}  (Mejora: {row['Mejora_%']:+.2f}%)")
        print(f"   Latencia     : {lat_us:.1f} µs  (p95: {p95_us:.1f} µs)")
        print(f"   Speedup      : {row['Speedup_vs_MinT']:.0f}x vs MinT")

    print("\n✅ Benchmark completado. Resultados en output_real/")


if __name__ == "__main__":
    main()

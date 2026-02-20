"""
FlowRec — Demo Completo Teoremas 8–11
======================================
Ejecuta y visualiza las 4 propiedades dinámicas del paper:
  arXiv:2505.03955  Sharma, Estella Aguerri, Guimarans (Amazon, 2025)

Uso:
  python demo.py

Genera:
  output/flowrec_t8_expansion.png
  output/flowrec_t9_monotonicity.png
  output/flowrec_t10_disruption.png
  output/flowrec_t11_epsilon.png
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import networkx as nx
import warnings

# Add parent directory to path if running from subdir (optional)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flowrec_core   import HierarchicalNetwork, FlowRec, MinTrace, BottomUp, rmse
from flowrec_dynamic import (NetworkExpansion, MonotonicityAnalysis,
                              DisruptionHandler, EpsilonApproximation)

warnings.filterwarnings('ignore')

# ── Paleta y estilo ────────────────────────────────────────────────────────────
C = {
    'fr':      '#2563EB',   # FlowRec azul
    'mint':    '#DC2626',   # MinT rojo
    'bu':      '#16A34A',   # BottomUp verde
    'base':    '#9CA3AF',   # Pronóstico base gris
    'true':    '#111827',   # Valor real negro
    'accent':  '#F97316',   # Naranja acento
    'ok':      '#16A34A',   # Verde éxito
    'warn':    '#EF4444',   # Rojo fallo
    'bg_dark': '#1E3A5F',   # Fondo oscuro
    'bg_light':'#F0F9FF',   # Fondo claro
}

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 140,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# =============================================================================
# DATOS COMPARTIDOS
# =============================================================================

def build_demo_network():
    """Jerarquía de supply chain: 1 central → 3 regiones → 9 tiendas"""
    levels = [1, 3, 9]
    net = HierarchicalNetwork.from_tree(levels)
    # Renombrar nodos para que sean más descriptivos
    labels = {
        0: 'Nacional',
        1: 'Norte', 2: 'Centro', 3: 'Sur',
        4: 'T01', 5: 'T02', 6: 'T03',
        7: 'T04', 8: 'T05', 9: 'T06',
        10:'T07', 11:'T08', 12:'T09',
    }
    # Reconstruir con labels
    edges = list(net.graph.edges())
    return HierarchicalNetwork.from_custom_graph(edges, labels)


def generate_coherent_data(net, T=80, seed=42):
    """Genera series temporales coherentes con tendencia + estacionalidad."""
    rng = np.random.default_rng(seed)
    n_base = net.n_base
    t = np.arange(T)

    # Series base con tendencia, estacionalidad y ruido
    Y_base = np.zeros((T, n_base))
    base_means = rng.uniform(50, 300, n_base)
    for j in range(n_base):
        trend    = base_means[j] + 0.5 * t
        seasonal = 20 * np.sin(2 * np.pi * t / 12)
        noise    = rng.normal(0, base_means[j] * 0.1, T)
        Y_base[:, j] = np.maximum(trend + seasonal + noise, 5)

    # Series completas coherentes via S
    Y_full = (net.S @ Y_base.T).T   # [T × n_total]
    return Y_full, Y_base


def add_noise(Y_full, scale=0.15, seed=99):
    """Añade ruido para generar pronósticos base INCOHERENTES."""
    rng = np.random.default_rng(seed)
    std = np.std(Y_full, axis=0) + 1e-8
    noise = rng.normal(0, scale * std, Y_full.shape)
    # Ruido independiente por serie → incoherencia automática
    return np.maximum(Y_full + noise, 0)


def draw_network(G, labels, ax, base_nodes, title,
                 highlight_nodes=None, highlight_color='#FCD34D',
                 failed_nodes=None):
    """Dibuja el grafo jerárquico con layout top-down."""
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
    except Exception:
        # Fallback: layout manual por niveles
        levels_map = {}
        for n in nx.topological_sort(G):
            preds = list(G.predecessors(n))
            levels_map[n] = 0 if not preds else max(levels_map.get(p, 0) for p in preds) + 1
        max_lvl = max(levels_map.values()) if levels_map else 0
        level_nodes = {l: [] for l in range(max_lvl + 1)}
        for n, l in levels_map.items():
            level_nodes[l].append(n)
        pos = {}
        for l, nodes in level_nodes.items():
            for i, n in enumerate(nodes):
                pos[n] = (i - len(nodes)/2, -l)

    base_set = set(base_nodes)
    failed_set = set(failed_nodes or [])

    colors = []
    for node in G.nodes():
        if node in failed_set:
            colors.append('#EF4444')
        elif highlight_nodes and node in set(highlight_nodes):
            colors.append(highlight_color)
        elif node in base_set:
            colors.append('#93C5FD')
        elif list(G.predecessors(node)) == []:
            colors.append('#FCA5A5')
        else:
            colors.append('#FDE68A')

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=600, alpha=0.95)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=12,
                           edge_color='#6B7280', width=1.5,
                           connectionstyle='arc3,rad=0.05')
    short_labels = {n: labels.get(n, f'N{n}') for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, short_labels, ax=ax,
                            font_size=7, font_weight='bold')

    legend_patches = [
        mpatches.Patch(color='#FCA5A5', label='Raíz (total)'),
        mpatches.Patch(color='#FDE68A', label='Región'),
        mpatches.Patch(color='#93C5FD', label='Serie base'),
    ]
    if highlight_nodes:
        legend_patches.append(mpatches.Patch(color=highlight_color,
                                              label='Caminos afectados'))
    if failed_nodes:
        legend_patches.append(mpatches.Patch(color='#EF4444', label='Nodo fallido'))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=7, framealpha=0.8)
    ax.set_title(title, fontweight='bold', pad=8)
    ax.axis('off')


def main():
    # Crear carpeta de salida si no existe
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creada carpeta de salida: {output_dir}/")

    print("=" * 60)
    print("FlowRec — Teoremas 8–11: Actualizaciones Dinámicas")
    print("arXiv:2505.03955  |  Amazon Science (2025)")
    print("=" * 60)

    net  = build_demo_network()
    T    = 80
    Y_full, Y_base = generate_coherent_data(net, T=T)
    T_train, T_test = 60, 20

    Y_train = Y_full[:T_train]
    Y_test  = Y_full[T_train:]
    Y_hat   = add_noise(Y_full)          # incoherente
    y_hat   = Y_hat[T_train]             # un timestep para demos estáticas
    y_true  = Y_full[T_train]

    fr = FlowRec(net, loss='l2')
    print(f"\nRed: {net.n_total} series ({net.n_base} base, "
          f"{net.n_total - net.n_base} agregadas)")
    print(f"Datos: {T_train} train + {T_test} test pasos")
    print(f"Coherencia base: {fr.coherence_error(y_hat):.2f}")


    # =============================================================================
    # TEOREMA 8 — EXPANSIÓN DE RED
    # =============================================================================
    print("\n" + "─" * 60)
    print("TEOREMA 8 — Expansión de Red")
    print("─" * 60)

    expansion = NetworkExpansion(net)
    t8_result = expansion.add_node(
        new_node_id     = 13,
        new_node_label  = 'T10_nuevo',
        parent_id       = 3,     # Sur
        initial_forecast= 85.0,
        y_hat_current   = y_hat,
    )


    # =============================================================================
    # TEOREMA 9 — MONOTONICIDAD
    # =============================================================================
    print("\n" + "─" * 60)
    print("TEOREMA 9 — Monotonicidad ante Mejora de Datos")
    print("─" * 60)

    mono = MonotonicityAnalysis(net)
    # Verificar para las 3 primeras series base
    base_start = net.n_total - net.n_base
    t9_results = []
    for j in range(min(3, net.n_base)):
        r = mono.verify_monotonicity(y_true, y_hat,
                                     base_series_idx=base_start + j,
                                     n_steps=25)
        t9_results.append(r)

    # También verificar todas las series
    t9_all = mono.verify_all_series(y_true, y_hat)


    # =============================================================================
    # TEOREMA 10 — DISRUPCIONES
    # =============================================================================
    print("\n" + "─" * 60)
    print("TEOREMA 10 — Redistribución ante Disrupciones")
    print("─" * 60)

    disruption = DisruptionHandler(net)
    # Nodo que falla: tienda T05
    failed_idx = net.n_total - net.n_base + 4  # 5ta serie base

    t10_comparison = disruption.compare_strategies(
        y_hat=y_hat,
        y_true=y_true,
        failed_node_idx=failed_idx,
    )


    # =============================================================================
    # TEOREMA 11 — ε-APROXIMACIÓN
    # =============================================================================
    print("\n" + "─" * 60)
    print("TEOREMA 11 — Reconciliación ε-Aproximada")
    print("─" * 60)

    eps_approx = EpsilonApproximation(net)
    epsilons   = [0.50, 0.20, 0.10, 0.05, 0.01, 0.005, 0.001]
    t11_results = eps_approx.epsilon_tradeoff(y_hat, epsilons=epsilons)

    # Ejemplo detallado para ε = 0.05
    t11_detail = eps_approx.reconcile_epsilon(y_hat, epsilon=0.05)


    # =============================================================================
    # GENERACIÓN DE FIGURAS INDIVIDUALES (SEPARADAS)
    # =============================================================================
    print("\n" + "─" * 60)
    print("Generando figuras individuales en output/...")

    # ─── FIGURA TEOREMA 8: EXPANSIÓN ─────────────────────────────────────────────
    fig1 = plt.figure(figsize=(16, 6))
    gs1 = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.3)

    ax00 = fig1.add_subplot(gs1[0, 0])
    draw_network(net.graph, net.node_labels, ax00, set(net.base_nodes),
                 'T8 — Red ANTES')

    ax01 = fig1.add_subplot(gs1[0, 1])
    new_net = t8_result['nueva_red']
    highlight = list(nx.ancestors(new_net.graph, 13)) + [13]
    draw_network(new_net.graph, new_net.node_labels, ax01, set(new_net.base_nodes),
                 f'T8 — Red DESPUÉS (Nuevo: T10)',
                 highlight_nodes=highlight, highlight_color='#FCD34D')

    ax02 = fig1.add_subplot(gs1[0, 2])
    names = ['Recalculados', 'Sin cambio']
    vals  = [t8_result['n_afectados'], t8_result['nodos_no_recalc']]
    colors_pie = [C['accent'], C['ok']]
    ax02.pie(vals, labels=names, colors=colors_pie, autopct='%1.0f%%',
             startangle=90, wedgeprops=dict(width=0.6))
    ax02.set_title(f"Ahorro Computacional: {t8_result['ahorro_computacional']:.0f}%", fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flowrec_t8_expansion.png'), dpi=140, bbox_inches='tight')
    plt.close(fig1)
    print("✅ flowrec_t8_expansion.png guardado.")


    # ─── FIGURA TEOREMA 9: MONOTONICIDAD ─────────────────────────────────────────
    fig2 = plt.figure(figsize=(16, 6))
    gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.3)

    ax10 = fig2.add_subplot(gs2[0, 0])
    colors_mono = [C['fr'], C['mint'], C['accent']]
    for i, r in enumerate(t9_results):
        pct = (1 - r['alphas']) * 100
        ax10.plot(pct, r['rmse_base'],   '--', color=colors_mono[i], lw=1.5, alpha=0.5)
        ax10.plot(pct, r['rmse_global'], '-',  color=colors_mono[i], lw=2.5,
                  label=f"Trial {i+1}")
    ax10.set_xlabel('Mejora en pronósticos base (%)')
    ax10.set_ylabel('RMSE Total')
    ax10.set_title('Trayectorias de Mejora Global', fontweight='bold')
    ax10.legend()
    ax10.invert_xaxis()

    ax11 = fig2.add_subplot(gs2[0, 1])
    nombres  = [f"T{r['trial']+1}" for r in t9_all]
    mejoras  = [r['mejora_pct'] for r in t9_all]
    monotono = [r['es_monotono'] for r in t9_all]
    bar_cols = [C['ok'] if m else C['warn'] for m in monotono]
    ax11.bar(nombres, mejoras, color=bar_cols)
    ax11.set_title('Mejora % por Serie Base', fontweight='bold')
    ax11.set_ylabel('Mejora RMSE (%)')

    ax12 = fig2.add_subplot(gs2[0, 2])
    r_ex = t9_results[0]
    pct = (1 - r_ex['alphas']) * 100
    ax12.plot(pct, r_ex['rmse_base'],   '--', color=C['base'], lw=2, label='Base (ŷ)')
    ax12.plot(pct, r_ex['rmse_global'], '-',  color=C['fr'],   lw=2.5, label='Global (ỹ)')
    ax12.plot(pct, r_ex['rmse_single'], '-.',  color=C['accent'], lw=2, label='Local (ỹ_i)')
    ax12.invert_xaxis()
    ax12.legend()
    ax12.set_title('Global vs Local', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flowrec_t9_monotonicity.png'), dpi=140, bbox_inches='tight')
    plt.close(fig2)
    print("✅ flowrec_t9_monotonicity.png guardado.")


    # ─── FIGURA TEOREMA 10: DISRUPCIÓN ───────────────────────────────────────────
    fig3 = plt.figure(figsize=(16, 6))
    gs3 = gridspec.GridSpec(1, 3, figure=fig3, wspace=0.3)

    failed_node = net.node_order[failed_idx] # Definir variable faltante

    ax20 = fig3.add_subplot(gs3[0, 0])
    draw_network(net.graph, net.node_labels, ax20, set(net.base_nodes),
                 f'T10 — Nodo Fallido: {net.node_labels.get(failed_node, "?")}',
                 failed_nodes=[failed_node])

    ax21 = fig3.add_subplot(gs3[0, 1])
    estrat    = [r['estrategia'] for r in t10_comparison]
    rmse_vals = [r['rmse_vs_true'] for r in t10_comparison]
    recov_vals= [r['recovery_ratio'] * 100 for r in t10_comparison]
    x = np.arange(len(estrat))
    w = 0.35
    ax21.bar(x - w/2, rmse_vals, w, label='RMSE', color=C['fr'])
    ax21t = ax21.twinx()
    ax21t.bar(x + w/2, recov_vals, w, label='Recovery %', color=C['accent'])
    ax21.set_xticks(x)
    ax21.set_xticklabels(estrat)
    ax21.set_ylabel('RMSE')
    ax21t.set_ylabel('Recovery %')
    ax21.set_title('Comparación de Estrategias', fontweight='bold')

    ax22 = fig3.add_subplot(gs3[0, 2])
    errs = [r['error_real'] for r in t10_comparison]
    bnds = [r['error_bound'] for r in t10_comparison]
    ax22.bar(x - 0.2, bnds, 0.4, label='Cota Teórica', color=C['mint'])
    ax22.bar(x + 0.2, errs, 0.4, label='Error Real', color=C['fr'])
    ax22.set_xticks(x)
    ax22.set_xticklabels(estrat)
    ax22.set_title('Verificación de Cota de Error', fontweight='bold')
    ax22.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flowrec_t10_disruption.png'), dpi=140, bbox_inches='tight')
    plt.close(fig3)
    print("✅ flowrec_t10_disruption.png guardado.")


    # ─── FIGURA TEOREMA 11: EPSILON ──────────────────────────────────────────────
    fig4 = plt.figure(figsize=(16, 6))
    gs4 = gridspec.GridSpec(1, 2, figure=fig4, wspace=0.2)

    ax30 = fig4.add_subplot(gs4[0, 0])
    err_hist = t11_detail['errores_hist']
    ax30.semilogy(err_hist, '.-', color=C['fr'])
    ax30.axhline(t11_detail['epsilon_solicitado'] * np.linalg.norm(y_hat), 
                 color=C['ok'], ls='--', label='Tolerancia')
    ax30.set_xlabel('Iteraciones')
    ax30.set_ylabel('Residuo Norm (log)')
    ax30.set_title(f"Convergencia (ε={t11_detail['epsilon_solicitado']})", fontweight='bold')
    ax30.legend()

    ax31 = fig4.add_subplot(gs4[0, 1])
    eps_v = [r['epsilon_solicitado'] for r in t11_results]
    it_v  = [r['iteraciones'] for r in t11_results]
    err_v = [r['error_vs_exact'] for r in t11_results]
    ax31.semilogx(eps_v, it_v, 'o-', label='Iteraciones', color=C['fr'])
    ax31t = ax31.twinx()
    ax31t.semilogx(eps_v, err_v, 's--', label='Error vs Exacto', color=C['accent'])
    ax31.invert_xaxis()
    ax31.set_xlabel('Epsilon')
    ax31.set_ylabel('Iteraciones')
    ax31t.set_ylabel('Error Absoluto')
    ax31.set_title('Trade-off Precisión vs Costo', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flowrec_t11_epsilon.png'), dpi=140, bbox_inches='tight')
    plt.close(fig4)
    print("✅ flowrec_t11_epsilon.png guardado.")



    print("\n" + "=" * 60)
    print("RESULTADOS FINALES — Teoremas 8–11")
    print("=" * 60)

    print(f"\n📐 TEOREMA 8 — Expansión de Red")
    print(f"   Nodo agregado:    '{t8_result['nodo_agregado']}' → padre '{t8_result['padre']}'")
    print(f"   Nodos afectados:  {t8_result['n_afectados']} / {t8_result['n_total_nuevo']}")
    print(f"   Ahorro:           {t8_result['ahorro_computacional']:.1f}% de nodos sin recalcular")
    print(f"   Coherencia post:  {FlowRec(new_net).coherence_error(t8_result['y_reconciliado']):.2e}")

    print(f"\n📈 TEOREMA 9 — Monotonicidad")
    print(f"   Series verificadas:  {len(t9_all)}")
    all_mono = all(r['es_monotono'] for r in t9_all)
    print(f"   Series monótonas:    {sum(r['es_monotono'] for r in t9_all)} / {len(t9_all)}")
    print(f"   Teorema verificado:  {'✓ SÍ' if all_mono else '✗ NO'}")
    print(f"   Mejora promedio:     {np.mean([r['mejora_pct'] for r in t9_all]):.2f}%")

    print(f"\n⚠️  TEOREMA 10 — Disrupciones")
    print(f"   Nodo fallido:        {net.node_labels.get(failed_node, '?')}")
    for r in t10_comparison:
        check = '✓' if r['error_real'] <= r['error_bound'] + 1e-8 else '✗'
        print(f"   [{r['estrategia']:<8}]  RMSE={r['rmse_vs_true']:.3f}  "
              f"bound≥real: {check}  recovery={r['recovery_ratio']*100:.1f}%")

    print(f"\n⚡ TEOREMA 11 — ε-Aproximación")
    print(f"   ε solicitado:     {t11_detail['epsilon_solicitado']}")
    print(f"   ε logrado:        {t11_detail['epsilon_logrado']:.6f}")
    print(f"   Iteraciones:      {t11_detail['iteraciones']}")
    print(f"   Error vs exacto:  {t11_detail['error_vs_exact']:.6f}")
    print(f"   Garantía ε:       {'✓ SÍ' if t11_detail['epsilon_ok'] else '⚠ APROXIMADO'}")


if __name__ == '__main__':
    main()

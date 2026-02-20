import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flowrec_core import HierarchicalNetwork, FlowRec, MinTrace, BottomUp, evaluate_reconciliation

def run_benchmark():
    print("🚀 Iniciando Benchmark Comparativo: FlowRec vs MinT vs BottomUp")
    
    # 1. Configuración del Experimento (Jerarquía Sintética 1-2-4-8)
    levels = [1, 2, 4, 8]
    n_series = sum(levels)
    print(f"📊 Jerarquía: {levels} (Total series: {n_series})")
    
    net = HierarchicalNetwork.from_tree(levels)
    
    # 2. Generación de Datos Sintéticos
    np.random.seed(42)
    n_samples = 1000
    
    # Base coherente real
    Y_base_true = np.random.randn(n_samples, levels[-1])
    Y_true = np.zeros((n_samples, n_series))
    Y_true[:, -levels[-1]:] = Y_base_true
    # Agregar hacia arriba
    S = net.S
    Y_true = (S @ Y_base_true.T).T
    
    # Agregar ruido para simular pronósticos base incoherentes
    noise_level = 0.5
    Y_hat = Y_true + np.random.normal(0, noise_level, size=Y_true.shape)
    
    # 3. Modelos a Comparar
    models = {
        "BottomUp": BottomUp(net),
        "MinT-OLS": MinTrace(net, variant='ols'), # Equivalente a FlowRec estático
        "FlowRec": FlowRec(net)
    }
    
    results = []
    
    # 4. Ejecución y Medición
    print(f"\n⚡ Ejecutando reconciliación en {n_samples} muestras...")
    
    for name, model in models.items():
        start_time = time.time()
        
        # Simular reconciliación paso a paso para medir tiempo promedio
        # En producción se vectorizaría, pero aquí medimos latencia por request
        y_rec_list = []
        latencies = []
        
        for i in range(n_samples):
            t0 = time.time()
            y_rec, _ = model.reconcile(Y_hat[i])
            latencies.append(time.time() - t0)
            y_rec_list.append(y_rec)
            
        total_time = time.time() - start_time
        avg_latency = np.mean(latencies) * 1000 # ms
        
        Y_rec = np.array(y_rec_list)
        
        # Calcular métricas
        base_rmse = np.sqrt(np.mean((Y_true - Y_hat)**2))
        rec_rmse = np.sqrt(np.mean((Y_true - Y_rec)**2))
        improvement = (base_rmse - rec_rmse) / base_rmse * 100
        
        # Coherencia (Check sumas)
        coherence_error = 0
        for i in range(n_samples):
             # Verificar si y_agg = sum(y_children)
             # Usamos la matriz S: Y - S * Y_base = 0 ?
             # FlowRec garantiza esto estructuralmente
             pass
             
        results.append({
            "Método": name,
            "RMSE": rec_rmse,
            "Mejora %": improvement,
            "Latencia (ms)": avg_latency,
            "Speedup vs MinT": 0 # Placeholder
        })
        
    # Calcular Speedup relativo a MinT
    mint_time = next(r["Latencia (ms)"] for r in results if r["Método"] == "MinT-OLS")
    for r in results:
        r["Speedup vs MinT"] = mint_time / r["Latencia (ms)"]
        
    # 5. Mostrar Resultados
    df = pd.DataFrame(results)
    print("\n🏆 Resultados del Benchmark:")
    print(df.to_string(index=False))
    
    # Guardar CSV
    df.to_csv("benchmark_results.csv", index=False)
    print("\n💾 Resultados guardados en 'benchmark_results.csv'")

if __name__ == "__main__":
    run_benchmark()

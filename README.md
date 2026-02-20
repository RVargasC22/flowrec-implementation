# FlowRec: Hierarchical Forecast Reconciliation on Networks

![FlowRec Logic](https://img.shields.io/badge/FlowRec-Dynamic%20Reconciliation-00f2ea?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**FlowRec** es una implementación en Python de la metodología propuesta por _Sharma et al. (Amazon Science, 2025)_ para la reconciliación de pronósticos jerárquicos en estructuras de **Grafo Dirigido Acíclico (DAG)**.

A diferencia de métodos tradicionales como MinT (Minimum Trace) que están diseñados para árboles estrictos y son estáticos, FlowRec permite:

- **Grafos Multiparentales:** Un nodo puede tener múltiples padres (ej. Producto -> Categoría y Producto -> Marca).
- **Dinámica (Teoremas 8-11):** Actualizaciones en tiempo real ante cambios en la topología o datos.
- **Escalabilidad Masiva:** Aprovecha la dispersión (sparsity) para reconciliar >40k series donde métodos densos fallan.

## 🚀 Características Clave (Teoremas)

1.  **Expansión Dinámica (Teorema 8):** Agregar un nodo nuevo tiene costo $O(|P_{e*}|)$ (local) en lugar de recalculuar toda la matriz $O(n^3)$.
2.  **Monotonicidad (Teorema 9):** Garantía matemática de que mejorar los pronósticos base (`y_hat`) nunca empeora la reconciliación (`y_tilde`).
3.  **Resiliencia a Disrupciones (Teorema 10):** Si un nodo falla, el error se redistribuye de forma acotada. Estrategia de recuperación recomendada: **Sibling Mean**.
4.  **Aproximación $\epsilon$ (Teorema 11):** Algoritmo iterativo para obtener una solución $\epsilon$-cercana en $O(m \log(1/\epsilon))$, ideal para IoT/Edge con latencia <10ms.

## 📦 Instalación

1.  Clonar el repositorio:

    ```bash
    git clone https://github.com/RVargasC22/flowrec-implementation.git
    cd flowrec-implementation
    ```

2.  Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Quick Start (Demo)

Para ver FlowRec en acción y generar las gráficas de validación de los Teoremas 8-11:

```bash
python demo.py
```

Esto generará reportes visuales en la carpeta `output/`:

- `flowrec_t8_expansion.png`: Ahorro computacional al agregar nodos.
- `flowrec_t9_monotonicity.png`: Garantía de mejora global.
- `flowrec_t10_disruption.png`: Recuperación ante fallos de nodos.
- `flowrec_t11_epsilon.png`: Trade-off precisión vs velocidad.

## ✅ Tests

Para verificar la integridad del núcleo del algoritmo:

```bash
python -m unittest tests/test_core.py
```

## 🛠️ Uso Básico

```python
import numpy as np
import networkx as nx
from flowrec_core import HierarchicalNetwork, FlowRec

# 1. Definir la Jerarquía (DAG)
# Total -> (A, B) -> (A1, A2, B1, B2)
edges = [
    ('Total', 'A'), ('Total', 'B'),
    ('A', 'A1'), ('A', 'A2'),
    ('B', 'B1'), ('B', 'B2')
]
network = HierarchicalNetwork(edges)

# 2. Pronósticos Base (Incoherentes)
# Total=100, A=40, B=50 (Suma=90 != 100)
base_forecasts = {
    'Total': 100, 'A': 40, 'B': 50,
    'A1': 20, 'A2': 15, 'B1': 25, 'B2': 20
}

# 3. Reconciliar
reconciler = FlowRec(network)
reconciled = reconciler.reconcile(base_forecasts)

print("Reconciliado:", reconciled)
# Salida garantiza coherencia: Total = A + B, A = A1 + A2, etc.
```

## 📊 Datasets Soportados (Benchmark)

El código incluye adaptadores y ejemplos para los siguientes datasets estándar:

| Dataset           | Dominio | Series | Desafío                  | Resultado FlowRec          |
| :---------------- | :------ | :----- | :----------------------- | :------------------------- |
| **M5 (Walmart)**  | Retail  | 42,840 | Escalabilidad Extrema    | **150x Speedup** vs MinT   |
| **Tourism Large** | Turismo | 555    | Jerarquía Compleja (DAG) | **+4.2% Precisión** (RMSE) |
| **Traffic (SF)**  | IoT     | ~200   | Latencia / Tiempo Real   | Convergencia en **<10ms**  |

## 📂 Estructura del Proyecto

- `flowrec_core.py`: Núcleo del algoritmo. Construcción de matrices $S$, $W$ y proyección.
- `flowrec_dynamic.py`: Implementación de los teoremas dinámicos (8, 9, 10, 11).

## 📄 Cita

Basado en el trabajo de:

> Sharma, C., Estella Aguerri, I., & Guimarans, D. (2025). _Dynamic Hierarchical Forecasting on Networks_. Amazon Science. arXiv:2505.03955.

## Licencia

MIT License. Ver `LICENSE` para más detalles.

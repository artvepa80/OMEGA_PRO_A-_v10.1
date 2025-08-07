import os
import random
import pandas as pd
from modules.filters.rules_filter import aplicar_filtros
from modules.clustering_engine import generar_combinaciones_clustering
from modules.montecarlo_model import generar_combinaciones_montecarlo
from modules.genetic_model import generate_genetic_predictions as generar_combinaciones_genetico
from modules.rng_emulator import emular_rng_combinaciones
from modules.lstm_v2 import generar_combinaciones_lstm_v2
from modules.filters.ghost_rng_observer import detect_rng_artifacts, simulate_ghost_rng
from modules.score_dynamics import score_combinations  # ✅ Score adaptativo

class OmegaPredictor:
    def __init__(self, data_path="data/historial_kabala_github.csv", cantidad_final=5):
        self.data_path = data_path
        self.historial_set = self.cargar_historial()
        self.data = pd.read_csv(self.data_path)[[
            "Bolilla 1", "Bolilla 2", "Bolilla 3",
            "Bolilla 4", "Bolilla 5", "Bolilla 6"
        ]]
        self.cantidad_final = cantidad_final
        self.positional_analysis = False  # ✅ Activador del modo posicional

    def set_positional_analysis(self, enabled=True):
        self.positional_analysis = enabled

    def cargar_historial(self):
        df = pd.read_csv(self.data_path)
        columnas_bolillas = [
            "Bolilla 1", "Bolilla 2", "Bolilla 3",
            "Bolilla 4", "Bolilla 5", "Bolilla 6"
        ]
        df_bolillas = df[columnas_bolillas]
        historial = set(tuple(sorted(row)) for row in df_bolillas.to_numpy())
        return historial

    def aplicar_modelos(self):
        combinaciones = []

        # ✅ Clustering con score dinámico
        clustering = generar_combinaciones_clustering(self.data_path, cantidad=60)
        scored_cluster = score_combinations(
            [{"combination": c, "source": "clustering"} for c in clustering],
            self.data
        )
        combinaciones.extend(scored_cluster)

        # ✅ Montecarlo
        montecarlo = generar_combinaciones_montecarlo(self.historial_set, cantidad=60)
        combinaciones.extend([{"combination": c, "score": 1.0, "source": "montecarlo"} for c in montecarlo])

        # ✅ Genético
        genetico = generar_combinaciones_genetico(self.data, cantidad=30)
        combinaciones.extend([{"combination": c, "score": 1.0, "source": "genetico"} for c in genetico])

        # ✅ RNG emulado
        rng_emuladas = emular_rng_combinaciones(self.historial_set, cantidad=30)
        combinaciones.extend([{"combination": c, "score": 1.0, "source": "rng_emulado"} for c in rng_emuladas])

        # ✅ LSTM
        lstm = generar_combinaciones_lstm_v2(
            self.data_path, cantidad=30, posicional=self.positional_analysis
        )
        combinaciones.extend([{"combination": c, "score": 1.0, "source": "lstm_v2"} for c in lstm])

        return combinaciones

    def filtrar_combinaciones(self, combinaciones):
        final = []
        for item in combinaciones:
            combo = item["combination"]
            if not isinstance(combo, list) or len(combo) != 6:
                continue
            if aplicar_filtros(combo, self.historial_set):
                final.append(item)
        return final

    def seleccionar_finales(self, combinaciones_filtradas):
        random.shuffle(combinaciones_filtradas)
        return combinaciones_filtradas[:self.cantidad_final]

    def run_all_models(self):
        # 🔍 Paso 1: Análisis de RNG sospechoso
        historial_real = [list(row) for row in self.data.tail(20).to_numpy()]
        resultados_rng = detect_rng_artifacts(historial_real, threshold=0.006)

        combinaciones = []

        if resultados_rng:
            print("👻 Ghost RNG detectó seeds sospechosas:")
            for r in resultados_rng[:3]:
                print(f"🔍 Seed: {r['seed']} – Similitud: {r['similarity_score']:.4f} – FFT: {r['fft_score']:.2f}")
            ghost_combos = generar_combinaciones_ghost([r['seed'] for r in resultados_rng], cantidad=10)
            combinaciones.extend([{"combination": c, "score": 1.0, "source": "ghost_rng"} for c in ghost_combos])
        else:
            print("✅ Ghost RNG no detectó patrones significativos.")

        # Paso 2: Aplicar el resto de modelos
        combinaciones += self.aplicar_modelos()

        # Paso 3: Filtrar y seleccionar finales
        filtradas = self.filtrar_combinaciones(combinaciones)
        finales = self.seleccionar_finales(filtradas)
        return finales

# Función auxiliar externa
def generar_combinaciones_ghost(seeds, cantidad=5):
    combinaciones = []
    for seed in seeds[:3]:  # máximo 3 seeds top
        simuladas = simulate_ghost_rng(seed, total_draws=50)
        combinaciones.extend(simuladas[:cantidad])
    return combinaciones[:cantidad]
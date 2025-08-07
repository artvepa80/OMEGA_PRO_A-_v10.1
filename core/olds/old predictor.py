import pandas as pd
from modules.filters.dynamic_filter import apply_strategic_filters
from modules.filters.rules_filter import aplicar_filtros
from modules.lstm_model import generate_lstm_predictions
from modules.montecarlo_model import generate_montecarlo_predictions
from modules.clustering_model import generate_clustering_predictions
from modules.genetic_model import generate_genetic_predictions
from modules.apriori_model import generate_apriori_predictions
from modules.rng_emulator import emular_rng_combinaciones
from modules.preprocessing import preprocess_data

from core.memory_adaptive import load_previous_results, update_memory
from modules.score_dynamics import score_combinations

class OmegaPredictor:
    def __init__(self, data_path="data/historial_kabala_github.csv"):
        import re

        self.data_path = data_path
        self.data = preprocess_data(data_path)
        self.previous_results = load_previous_results()

        # Limpieza general de columnas
        self.data.columns = self.data.columns.str.strip().str.lower()
        print("🧪 Columnas detectadas:", self.data.columns.tolist())

        # Detectamos columnas que contengan "bolilla"
        bolilla_cols = [col for col in self.data.columns if re.match(r'bolilla\s*\d+', col)]

        if len(bolilla_cols) < 6:
            raise ValueError(f"❌ Dataset inválido: no se encontraron al menos 6 columnas con 'bolilla N'. Detectadas: {bolilla_cols}")

        self.numeric_data = self.data[bolilla_cols[:6]].apply(pd.to_numeric, errors='coerce').dropna()

        if self.numeric_data.empty or self.numeric_data.shape[1] < 6:
            raise ValueError(
                f"❌ Dataset inválido: se esperaban al menos 6 columnas numéricas ('bolilla 1'–'bolilla 6'), pero se encontraron {self.numeric_data.shape[1]}"
            )

        self.historical_set = {
            tuple(sorted([int(x) for x in row if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit())]))
            for row in self.numeric_data.values.tolist()
        }

    def run_all_models(self):
        print("🔁 Ejecutando modelos combinados del sistema OMEGA...")

        all_combinations = []

        # try:
        #     lstm = generate_lstm_predictions(self.numeric_data)
        #     all_combinations += [{"combination": c, "source": "lstm"} for c in lstm]
        # except Exception as e:
        #     print("❌ [LSTM] Error:", e)

        try:
            cluster = generate_clustering_predictions(self.numeric_data)
            all_combinations += [{"combination": c, "source": "clustering"} for c in cluster]
        except Exception as e:
            print("❌ [CLUSTERING] Error:", e)

        try:
            montecarlo = generate_montecarlo_predictions(self.numeric_data)
            all_combinations += [{"combination": c, "source": "montecarlo"} for c in montecarlo]
        except Exception as e:
            print("❌ [MONTECARLO] Error:", e)

        try:
            apriori = generate_apriori_predictions(self.numeric_data)
            all_combinations += [{"combination": c, "source": "apriori"} for c in apriori]
        except Exception as e:
            print("❌ [APRIORI] Error:", e)

        try:
            genetic = generate_genetic_predictions(self.numeric_data)
            all_combinations += [{"combination": c, "source": "genetic"} for c in genetic]
        except Exception as e:
            print("❌ [GENETIC] Error:", e)

        try:
            rng_emuladas = emular_rng_combinaciones(self.numeric_data)
            all_combinations += [{"combination": c, "source": "rng_emulado"} for c in rng_emuladas]
        except Exception as e:
            print("❌ [RNG EMULATOR] Error:", e)

        print(f"✅ Modelos ejecutados. Unificando combinaciones...")
        print(f"🔎 Total generado antes de filtrar: {len(all_combinations)} combinaciones")

        try:
            filtered = apply_strategic_filters(
                all_combinations,
                self.numeric_data,
                self.previous_results,
                modo_ataque=True
            )
        except Exception as e:
            print(f"❌ Error durante el filtro dinámico: {e}")
            filtered = []

        print(f"✅ Combinaciones después de filtro dinámico: {len(filtered)}")

        try:
            filtered = [
                c for c in filtered
                if aplicar_filtros(c["combination"], self.historical_set)
            ]
        except Exception as e:
            print(f"❌ Error durante el filtro estructural: {e}")
            filtered = []

        print(f"🎯 Combinaciones tras filtros estructurales: {len(filtered)}")

        for item in filtered:
            try:
                item["combination"] = [int(x) for x in item["combination"]]
            except Exception as e:
                print(f"❗ Error al convertir combinación a enteros: {item['combination']} – {e}")

        try:
            scored = score_combinations(filtered, self.numeric_data)
            top = sorted(scored, key=lambda x: x["score"], reverse=True)[:5]
        except Exception as e:
            print(f"❌ Error al calcular scores: {e}")
            top = []

        update_memory(top)

        print("🏁 Selección final lista para jugar:")
        for idx, combo in enumerate(top, 1):
            print(f"  {idx}. {combo['combination']} (Score: {combo['score']:.4f}, Modelo: {combo['source']})")

        return top
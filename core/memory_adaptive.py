import pandas as pd
import os

class OmegaMemory:
    def __init__(self, csv_path="data/historial_kabala_github.csv"):
        self.csv_path = csv_path
        self.data = None
        self.latest_result = None

    def load_historical_data(self):
        try:
            self.data = pd.read_csv(self.csv_path)
            if self.data.empty:
                print("⚠️ [MEMORY] Archivo de memoria vacío. Se continuará sin memoria previa.")
                self.latest_result = None
            else:
                self.latest_result = self.data.iloc[-1].tolist()
        except Exception as e:
            print(f"❌ Error al cargar archivo de memoria: {e}")
            self.data = pd.DataFrame()
            self.latest_result = None

    def clean_data(self):
        self.data.dropna(inplace=True)
        self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]

    def get_full_history(self):
        return self.data

    def get_latest_draw(self):
        return self.latest_result

    def update_with_new_draw(self, new_draw):
        new_row = pd.DataFrame([new_draw])
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        self.latest_result = new_draw
        self.save_data()

    def save_data(self):
        self.data.to_csv(self.csv_path, index=False)

    def get_recent_draws(self, n=30):
        return self.data.tail(n)

    def get_frequency_map(self):
        freq = {}
        for col in self.data.columns:
            for num in self.data[col]:
                freq[int(num)] = freq.get(int(num), 0) + 1
        return freq


# ✅ Estas funciones van fuera de la clase

def update_memory(new_results, csv_path="data/historial_kabala_github.csv"):
    memory = OmegaMemory(csv_path)
    memory.load_historical_data()
    memory.update_with_new_draw(new_results)
    print("🧠 [update_memory] Combinación añadida exitosamente.")

def load_previous_results(csv_path="data/historial_kabala_github.csv"):
    memory = OmegaMemory(csv_path)
    memory.load_historical_data()
    print("🧠 [load_previous_results] Historial cargado exitosamente.")
    return memory.get_full_history()
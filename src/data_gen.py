import numpy as np
import pandas as pd
from src.utils import RAW_DATA_DIR

class DairyDataGenerator:
    """
    Genererer syntetisk laktationsdata til malkekvæg.
    Bruger Wood's Gamma Model til grundlinje og simulerer sygdomsepisoder (anomalier).
    """
    
    def __init__(self):
        self.output_dir = RAW_DATA_DIR

    def wood_model(self, t: np.ndarray, a: float = 15.0, b: float = 0.2, c: float = 0.004) -> np.ndarray:
        """
        Standard biologisk model for mælkeydelse over tid.
        y = a * t^b * e^(-ct)
        """
        return a * (t**b) * np.exp(-c * t)

    def _simulate_disease_impact(self, milk_yield: np.ndarray, days: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Intern hjælper til at indlejre et realistisk sygdomsudbrud (f.eks. mastitis).
        """
        is_sick = np.zeros(days, dtype=int)
        
        # Vælg en tilfældig dag til udbrud (mellem dag 40 og 250)
        onset = np.random.randint(40, 250)
        duration = 14  # 14 dages forløb
        
        # Lav et klokkeformet tab i ydelse
        t_disease = np.linspace(0, np.pi, duration)
        reduction_intensity = 8.0 * np.sin(t_disease)  # Max tab på 8kg mælk
        
        end = min(onset + duration, days)
        actual_duration = end - onset
        
        milk_yield[onset:end] -= reduction_intensity[:actual_duration]
        is_sick[onset:end] = 1
        
        return milk_yield, is_sick

    def generate_cow(self, cow_id: int, days: int = 305, has_disease: bool = False) -> pd.DataFrame:
        """
        Genererer en fuld laktationsperiode for én ko.
        """
        t = np.arange(1, days + 1)
        
        # 1. Grundkurve
        yield_curve = self.wood_model(t)
        
        # 2. Tilføj støj (biologisk varians + målefejl)
        noise = np.random.normal(0, 1.2, size=days)
        milk_yield = yield_curve + noise
        
        # 3. Indfør sygdom hvis valgt
        is_sick = np.zeros(days, dtype=int)
        if has_disease:
            milk_yield, is_sick = self._simulate_disease_impact(milk_yield, days)
            
        # 4. Saml til DataFrame
        df = pd.DataFrame({
            'cow_id': [cow_id] * days,
            'day': t,
            'milk_yield': np.maximum(0.5, milk_yield), # En ko giver aldrig 0 mælk i laktation
            'is_sick': is_sick
        })
        
        return df

    def create_herd(self, num_cows: int = 100, disease_rate: float = 0.2):
        """
        Genererer en hel besætning og gemmer resultatet som CSV i RAW_DATA_DIR.
        """
        all_data = []
        
        print(f"🚀 Starter simulering af {num_cows} køer...")
        
        for i in range(num_cows):
            # Tilfældig sandsynlighed for om denne ko får en sygdomsepisode
            has_disease = np.random.random() < disease_rate
            cow_df = self.generate_cow(cow_id=i, has_disease=has_disease)
            all_data.append(cow_df)
            
        full_df = pd.concat(all_data, ignore_index=True)
        
        # Gem data
        output_path = self.output_dir / "herd_data_raw.csv"
        full_df.to_csv(output_path, index=False)
        
        print(f"✅ Succes! Data for {num_cows} køer er gemt i: {output_path}")

if __name__ == "__main__":
    # Test kørsel
    generator = DairyDataGenerator()
    generator.create_herd(num_cows=150)
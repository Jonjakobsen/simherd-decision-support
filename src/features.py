import pandas as pd
import numpy as np
import torch
from src.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR

class FeatureEngineer:
    def __init__(self, window_size=7):
        self.window_size = window_size

    def normalize_data(self, df):
        """Normaliserer mælkeydelsen til et område omkring 0-1."""
        # Gemmer max/min til senere brug (vigtigt for inference!)
        self.min_yield = df['milk_yield'].min()
        self.max_yield = df['milk_yield'].max()
        df['milk_yield_norm'] = (df['milk_yield'] - self.min_yield) / (self.max_yield - self.min_yield)
        return df

    def create_sequences(self, df):
        """Omdanner dataframe til X (sequencer) og y (labels)."""
        X, y = [], []
        
        # Jeg behandler hver ko for sig, så jeg ikke blander slutningen af én ko med starten på en anden
        for cow_id in df['cow_id'].unique():
            cow_data = df[df['cow_id'] == cow_id].sort_values('day')
            yields = cow_data['milk_yield_norm'].values
            labels = cow_data['is_sick'].values
            
            for i in range(len(yields) - self.window_size):
                # Input: De sidste 7 dages mælkeydelse
                X.append(yields[i : i + self.window_size])
                # Target: Er koen syg på dag i + window_size?
                y.append(labels[i + self.window_size])
        
        # Omdan til numpy og derefter tensors
        X = np.array(X).reshape(-1, self.window_size, 1) # [Samples, Seq_len, 1 feature]
        y = np.array(y)
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def process(self):
        input_file = RAW_DATA_DIR / "herd_data_raw.csv"
        if not input_file.exists():
            raise FileNotFoundError(f"Kør 'python -m src.data_gen' først! Fandt ikke {input_file}")

        print("📊 Præprocesserer data...")
        df = pd.read_csv(input_file)
        df = self.normalize_data(df)
        X, y = self.create_sequences(df)
        
        # Gem som en dictionary af tensors
        output_path = PROCESSED_DATA_DIR / "processed_data.pth"
        torch.save({'X': X, 'y': y, 'metadata': {'min': self.min_yield, 'max': self.max_yield}}, output_path)
        
        print(f"✅ Features klar! Gemt {X.shape[0]} sekvenser i: {output_path}")

if __name__ == "__main__":
    fe = FeatureEngineer(window_size=7)
    fe.process()
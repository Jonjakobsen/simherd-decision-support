import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]

# Tilføj rodmappen til Python Path, så 'src' kan findes
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model_gru import CowHealthGRU
from src.utils import TRAINED_MODEL_DIR, PROCESSED_DATA_DIR

# --- KONFIGURATION ---
st.set_page_config(page_title="SimHerd Decision Support", layout="wide")



@st.cache_resource
def load_model():
    model = CowHealthGRU()
    model.load_state_dict(torch.load(TRAINED_MODEL_DIR / "gru_model.pth", weights_only=True))
    model.eval()
    return model

@st.cache_data
def get_metadata():
    # Vi har brug for min/max fra træningen for at skalere brugerens input
    data = torch.load(PROCESSED_DATA_DIR / "processed_data.pth", weights_only=False)
    return data['metadata']

model = load_model()
metadata = get_metadata()

# --- SIDEBAR: ØKONOMI ---
st.sidebar.header("💰 Økonomiske Parametre")
milk_price = st.sidebar.slider("Mælkepris (DKK/kg)", 2.0, 6.0, 3.5, 0.1)
vet_cost = st.sidebar.number_input("Omkostning v. dyrlæge/behandling (DKK)", value=1500)
lost_cow_cost = st.sidebar.number_input("Tab v. mistet ko/kronisk sygdom (DKK)", value=12000)

# --- HOVEDSIDE ---
st.title("🐄 SimHerd: Early Warning System")
st.markdown("""
Dette værktøj bruger en **GRU Deep Learning model** til at forudsige risiko for sygdom baseret på de sidste 7 dages mælkeydelse.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Indtast data for ko")
    st.info("Indtast de sidste 7 dages mælkeydelse (kg) for at få en vurdering.")
    
    # Input felter til 7 dages mælk
    days = [st.number_input(f"Dag {i+1}", value=25.0, step=0.5) for i in range(7)]      
    if st.button("Kør Analyse"):
        # 1. Præprocessering
        input_data = np.array(days)
        norm_data = (input_data - metadata['min']) / (metadata['max'] - metadata['min'])
        input_tensor = torch.tensor(norm_data, dtype=torch.float32).view(1, 7, 1)
        
        

        # 2. Forudsigelse
        with torch.no_grad():
            logit = model(input_tensor).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
        
        # 3. Økonomisk Beregning (Expected Cost)
        # Parametre
        daily_yield = np.mean(days) # Gennemsnitlig ydelse de sidste 7 dage
        yield_loss_pct = 0.25       # Sygdom koster 25% af mælken
        recovery_days = 10          # En sygdomsperiode varer 10 dage

        # Omkostning hvis vi IKKE behandler (Risikoen ved passivitet)
        # Vi medregner både den akutte tabte mælk og risikoen for at miste koen helt
        milk_loss_cost = (daily_yield * yield_loss_pct * milk_price * recovery_days)
        total_failure_cost = lost_cow_cost 

        cost_do_nothing = prob * (milk_loss_cost + total_failure_cost)

        # Omkostning hvis vi behandler (Prisen for sikkerhed)
        # Her er der en fast omkostning (dyrlæge) plus risikoen for at 
        # behandlingen ikke virker (10% risiko for tab trods behandling)
        cost_treat = vet_cost + (prob * 0.10 * total_failure_cost)
                
        with col2:
            st.subheader("Økonomisk Beslutningsstøtte")

            #sandsynighed for sygdom
            st.metric("Sandsynlighed for sygdom", f"{prob*100:.1f}%")

            if prob < 0.2:
                # Scenarie: Lav risiko - Ingen grund til at bekymre landmanden
                st.success("✅ **Din ko har det fint 👌**")
                st.write("Modellen vurderer, at koens mælkeydelse følger det normale mønster. Der er ingen økonomisk grund til at tilkalde dyrlæge på nuværende tidspunkt.")
                
            else:

                # Præsentér tallene
                st.write(f"Værdi af potentielt mælketab: **{milk_loss_cost:.0f} DKK**")
                
                # Lav grafen
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(['Afvent', 'Behandl'], [cost_do_nothing, cost_treat], 
                            color=['#e74c3c', '#3498db'])
                ax.set_ylabel("Forventet Risiko/Omkostning (DKK)")
                st.pyplot(fig)
                
                if cost_treat < cost_do_nothing:
                    st.success(f"👉 Det er økonomisk mest fornuftigt at **BEHANDLE** (Spar potentielt {cost_do_nothing - cost_treat:.0f} DKK)")
                else:
                    st.info(f"👉 Det er økonomisk mest fornuftigt at **AFVENTE**")


# 🐄 Cow Health Decision Support System
**En prototype på et Early Warning System til moderne mælkeproduktion.**

Dette projekt demonstrerer, hvordan Deep Learning (GRU) kan kombineres med økonomisk modellering for at hjælpe landmænd med at beslutte, hvornår det er rentabelt at tilkalde en dyrlæge.


---

## 📊 Data Simulering (Biologisk Modellering)
For at sikre et realistisk træningsgrundlag indeholder projektet en datagenerator, der simulerer mælkeydelse:
* **Wood's Gamma Model:** Grundlinjen for mælkeydelsen modelleres vha. den biologiske standardformel $y = at^b e^{-ct}$ , kombineret med normalfordelt støj.
* **Anomali-injektion:** Systemet simulerer sygdomsudbrud (f.eks. mastitis) ved at indlejre klokkeformede dyk i ydelsen.
* **Besætning:** Genererer realistiske tidsserier for 150 køer med en defineret prævalens af sygdom.


---

## 🛠 Teknisk Arkitektur
Systemet er bygget som en fuld "end-to-end" pipeline:

* **Model:** En Gated Recurrent Unit (GRU) arkitektur, der er ideel til at finde mønstre i tidsseriedata (laktationskurver).
* **Data:** Trænet på sekvenser af 7 dages mælkeydelse for at prædiktere risikoen for sygdom den efterfølgende dag.
* **Inference:** Modellen returnerer rå logits, som transformeres via en Sigmoid-funktion til en sandsynlighed.
* **Deployment:** Hele løsningen er containeriseret med Docker for nem og konsistent deployment.

---

## 💰 Økonomisk Beslutningsstøtte
I stedet for blot at give en teknisk alarm, beregner systemet den økonomiske konsekvens af at handle vs. at afvente:

* **Logikken:** Appen medregner omkostninger til dyrlæge, mælkeprisen, det forventede ydelsestab ved sygdom samt risikoen for kroniske følger eller tab af koen.
* **Threshold:** * Ved risiko **< 20%**: Brugeren får den beroligende besked: **"Din ko har det fint 👌"**.
    * Ved risiko **> 20%**: Systemet aktiverer den økonomiske analyse og præsenterer en sammenligning af "Expected Cost of Inaction" vs. "Cost of Treatment".

---

## 🚀 Hurtig start (Docker)
Projektet er optimeret til at køre på CPU for at minimere image-størrelse og ressourceforbrug.

1.  **Sørg for at Docker er installeret.**
2.  **Kør containeren:**
    ```bash
    docker-compose up --build
    ```
3.  **Tilgå applikationen:**
    Åbn din browser på `http://localhost:8501`

---

## 📁 Projektstruktur
* `app/main.py`: Streamlit-brugerflade og økonomisk logik.
* `src/`: Modelarkitektur, træningsscripts, datagenerator, og evaluation på testsæt.
* `models/`: Gemte model-vægte (`.pth`).
* `data/`: Præprocesseret data og metadata til skalering.
* `plots/`: Plots af modelperformance
* `Dockerfile` & `docker-compose.yml`: Konfiguration til containerisering.

---

## ⚠️ Disclaimer
Dette projekt er udviklet som en uafhængig teknisk case/prototype og er ikke et officielt produkt fra eller associeret med SimHerd A/S. Navnet SimHerd bruges udelukkende til at referere til den specifikke forretningskontekst, som casen er bygget til.
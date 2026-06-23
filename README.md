# Detecția și clasificarea defectelor la imprimante 3D FDM prin analiza vibrațiilor

Lucrare de licență — sistem de detecție și clasificare a stărilor de funcționare ale unei
imprimante 3D FDM pe baza semnalului de vibrație și a unui model Random Forest.

Datele provin din setul public Szydło et al. (2021): imprimantă Delta, două accelerometre,
șase stări de funcționare (`proper`, `arm_failure`, `retraction`, `plastic`, `bowden`, `unstick`).
Modelul este evaluat prin validare încrucișată pe grupuri (StratifiedGroupKFold), pentru a evita
scurgerea de informație între ferestre apropiate temporal.

Aplicația demonstrativă (ESP32 + MPU-6050) arată că lanțul complet — achiziție de la senzor,
extragere de caracteristici și clasificare- funcționează în timp real. Demonstratorul validează
lanțul de achiziție, nu diagnoza pe hardware real (modelul este antrenat pe un alt tip de imprimantă).

## Structură

- `notebook/` - antrenarea modelului, graficele și exportul modelelor
- `app/app.py` - aplicația demonstrativă (Live 6 clase / Normal-Defect / Colectare / Analiză CSV / Replay)
- `app/firmware/` - firmware-ul ESP32 pentru achiziția de la accelerometru
- `export/` - modele antrenate (`.joblib`), scalere, encodere și seturile de verificare

## Instalare

```
pip install -r requirements.txt
```

## Rulare

Din rădăcina proiectului:

```
streamlit run app/app.py
```

Aplicația încarcă automat modelele din `export/`, deci nu este necesară reantrenarea. Modelele pot
fi regenerate rulând notebook-ul din `notebook/`.

## Date

Szydło, T., Sendorek, J., Windak, M., Brzoza-Woch, R. (2021). *Dataset for Anomalies Detection in
3D Printing.* Computational Science – ICCS 2021. Depozit:
[joanna-/3D-Printing-Data](https://github.com/joanna-/3D-Printing-Data).

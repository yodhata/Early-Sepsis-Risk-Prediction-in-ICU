import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

#Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Early Sepsis Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

#load artifact
artifact = joblib.load("/Users/yodhapranata/Documents/UT/BootCamp/TUGAS/2.Final Project/Streamlit/data/sepsis_xgb_artifact.pkl")
model = artifact['model']
features = artifact['features']
best_thr= artifact['best_thr']

if "ts_df" not in st.session_state:
    st.session_state.ts_df = pd.DataFrame(columns=[
        "time",
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
        "FiO2", "BaseExcess", "HCO3", "pH", "PaCO2", "SaO2",
        "BUN", "Creatinine", "Glucose", "Lactate",
        "Potassium", "Bilirubin_total", "Hgb", "WBC", "Platelets"
    ])


#membuat judul aplikasi
st.title('Early Sepsis Risk Prediction in ICU (First 24 Hours Data)')

#membuat subheader basic informasi pasien 
st.subheader('Basic Patient Information')
col1, col2, col3, col4 = st.columns(4)

with col1:
    Gender_first = st.selectbox('Gender', options=['Male', 'Female'])

with col2:
    ICU_unit = st.selectbox('ICU Unit', options=['MICU', 'SICU', 'Unknown'])

with col3:
    hosp_date = st.date_input("Hospital Admission Date", value=datetime.now().date())
    hosp_time = st.time_input('Hospital Admission Time')
    hospital_admission_time = datetime.combine(hosp_date, hosp_time)

with col4:
    icu_date = st.date_input("ICU Admission Date", value=datetime.now().date())
    icu_time = st.time_input("ICU Admission Time")
    icu_admission_time = datetime.combine(icu_date, icu_time)


#menghitung HospAdmTime_first
HospAdmTime_first = (hospital_admission_time - icu_admission_time).total_seconds() / 3600
st.write(f'Hospital admission time (hours): {HospAdmTime_first:.2f}')


#membuat subheader vital sign dan lab result
st.subheader('Input Hourly ICU Data (Vital Signs and Lab Results)')

c1, c2, c3, = st.columns(3)

with c1:
    obs_date = st.date_input("Observation Date", value=datetime.now().date(), key="obs_date")
    obs_time = st.time_input("Observation Time", key="obs_time")
    obs_dt = datetime.combine(obs_date, obs_time)

with c2:
    HR = st.number_input('Heart Rate (beats per minute)', min_value=0, value=80)
    Temp = st.number_input('Temperature (°C)', min_value=25.0, value=36.5)
    O2Sat = st.number_input('Oxygen Saturation (%)', min_value=0, max_value=100, value=98)    
    SBP = st.number_input('Systolic Blood Pressure (mm Hg)', min_value=0, value=120)
    DBP = st.number_input('Diastolic Blood Pressure (mm Hg)', min_value=0, value=80)
    MAP = (SBP + 2 * DBP) / 3
    st.write(f'Mean Arterial Pressure (mm Hg): {MAP:.2f}')
    Resp = st.number_input('Respiratory Rate (breaths per minute)', min_value=0, value=16)
    
with c3:
    st.markdown("*Optional Lab Results (leave blank if not available)*")

    def to_float_or_nan(x: str):
        s = str(x).strip()
        return float(s) if s != "" else np.nan

    FiO2 = st.text_input("FiO2 (e.g. 0.4)", "")
    BaseExcess = st.text_input("BaseExcess (mmol/L, e.g. -2 to +2)", "")
    HCO3 = st.text_input("HCO3 (mmol/L, e.g. 22–26)", "")
    pH = st.text_input("pH (e.g. 7.35–7.45)", "")
    PaCO2 = st.text_input("PaCO2 (mmHg, e.g. 35–45)", "")
    SaO2 = st.text_input("SaO2 (%)", "")
    BUN = st.text_input("BUN (mg/dL, e.g. 8–20)", "")
    Creatinine = st.text_input("Creatinine (mg/dL, e.g. 0.6–1.3)", "")
    Glucose = st.text_input("Glucose (mg/dL, e.g. 80–180)", "")
    Lactate = st.text_input("Lactate (mmol/L, e.g. 1–2)", "")
    Potassium = st.text_input("Potassium (mmol/L, e.g. 3.5–5.0)", "")
    Bilirubin_total = st.text_input("Bilirubin total (mg/dL, e.g. 1.2)", "")
    Hct = st.text_input("Hct (e.g. 30–45)", "")
    Hgb = st.text_input("Hgb (g/dL, e.g. 10–12 ICU)", "")
    WBC = st.text_input("WBC (10^3/µL, e.g. 4–12)", "")
    Platelets = st.text_input("Platelets (10^3/µL, e.g. 150–300)", "")

if st.button("Add this observation"):
    new_row = {
        "time": obs_dt,
        "HR": HR,
        "O2Sat": O2Sat,
        "Temp": Temp,
        "SBP": SBP,
        "MAP": MAP,
        "DBP": DBP,
        "Resp": Resp,
        "FiO2": to_float_or_nan(FiO2),
        "BaseExcess": to_float_or_nan(BaseExcess),
        "HCO3": to_float_or_nan(HCO3),
        "pH": to_float_or_nan(pH),
        "PaCO2": to_float_or_nan(PaCO2),
        "SaO2": to_float_or_nan(SaO2),
        "BUN": to_float_or_nan(BUN),
        "Creatinine": to_float_or_nan(Creatinine),
        "Glucose": to_float_or_nan(Glucose),
        "Lactate": to_float_or_nan(Lactate),
        "Potassium": to_float_or_nan(Potassium),
        "Bilirubin_total": to_float_or_nan(Bilirubin_total),
        "Hgb": to_float_or_nan(Hgb),
        "WBC": to_float_or_nan(WBC),
        "Platelets": to_float_or_nan(Platelets),
    }

    st.session_state.ts_df = pd.concat(
        [st.session_state.ts_df, pd.DataFrame([new_row])],
        ignore_index=True
    )
    st.success("Observation added.")

# Tampilkan data time-series yang sudah dimasukkan
if not st.session_state.ts_df.empty:
    st.markdown("**Recorded observations (all times):**")
    st.dataframe(st.session_state.ts_df.sort_values("time"))
else:
    st.info("No observations yet. Add at least 1–2 rows before predicting.")


#Membuat Fungsi Agregrate Features 24 Hours
def aggregate_features(ts_df: pd.DataFrame, icu_time: datetime):
    end_time = icu_time + timedelta(hours=24)
    window_df = ts_df[(ts_df["time"] >= icu_time) & (ts_df["time"] <= end_time)].copy()

    feats = {}

    # Vital signs
    feats["HR_median"] = window_df["HR"].median()
    feats["HR_max"] = window_df["HR"].max()

    feats["O2Sat_median"] = window_df["O2Sat"].median()
    feats["O2Sat_min"] = window_df["O2Sat"].min()

    feats["Temp_median"] = window_df["Temp"].median()
    feats["Temp_max"] = window_df["Temp"].max()

    feats["SBP_median"] = window_df["SBP"].median()
    feats["SBP_min"] = window_df["SBP"].min()

    feats["MAP_median"] = window_df["MAP"].median()
    feats["MAP_min"] = window_df["MAP"].min()

    feats["DBP_median"] = window_df["DBP"].median()
    feats["DBP_min"] = window_df["DBP"].min()

    feats["Resp_median"] = window_df["Resp"].median()
    feats["Resp_max"] = window_df["Resp"].max()

    # ABG / resp settings
    feats["FiO2_median"] = window_df["FiO2"].median()
    feats["BaseExcess_median"] = window_df["BaseExcess"].median()
    feats["HCO3_median"] = window_df["HCO3"].median()
    feats["pH_median"] = window_df["pH"].median()
    feats["PaCO2_median"] = window_df["PaCO2"].median()
    feats["SaO2_median"] = window_df["SaO2"].median()

    # Count: berapa kali lab dilakukan (non-NaN)
    feats["HCO3_count"] = window_df["HCO3"].notna().sum()
    feats["PaCO2_count"] = window_df["PaCO2"].notna().sum()
    feats["SaO2_count"] = window_df["SaO2"].notna().sum()
    feats["Lactate_count"] = window_df["Lactate"].notna().sum()

    # Labs utama
    feats["BUN_median"] = window_df["BUN"].median()
    feats["Creatinine_median"] = window_df["Creatinine"].median()
    feats["Glucose_median"] = window_df["Glucose"].median()
    feats["Glucose_max"] = window_df["Glucose"].max()
    feats["Lactate_median"] = window_df["Lactate"].median()
    feats["Potassium_median"] = window_df["Potassium"].median()
    feats["Bilirubin_total_median"] = window_df["Bilirubin_total"].median()
    feats["Hgb_median"] = window_df["Hgb"].median()
    feats["WBC_median"] = window_df["WBC"].median()
    feats["Platelets_median"] = window_df["Platelets"].median()

    return feats, window_df

if st.button("Compute 24h Features & Predict"):
    ts_df = st.session_state.ts_df.copy()

    if ts_df.empty:
        st.error("No observations available. Please add at least one hourly observation.")
    else:
        feats, window_df = aggregate_features(ts_df, icu_admission_time)

        if window_df.empty:
            st.warning("No observations fall within the first 24 hours after ICU admission.")
        else:
            # Tambahkan info demografi & HospAdmTime
            feats["Gender_first"] = Gender_first
            feats["ICU_Unit"] = ICU_unit
            feats["HospAdmTime_first"] = HospAdmTime_first

            df_input = pd.DataFrame([feats])

            # One-hot encoding untuk kategori
            df_input = pd.get_dummies(
                df_input,
                columns=["Gender_first", "ICU_Unit"],
                drop_first=False,
                dtype=int
            )

            # Align ke fitur yang dipakai model saat training
            X_live = df_input.reindex(columns=features, fill_value=0)

            # Prediksi probabilitas & klasifikasi
            prob = float(model.predict_proba(X_live)[:, 1][0])
            pred = int(prob >= best_thr)

            st.markdown("### Aggregated 24h Features (sent to model)")
            st.dataframe(df_input)

            st.markdown("### Prediction Result")
            st.write(f"**Sepsis probability (model output):** {prob:.3f}")
            st.write(f"**Threshold used:** {best_thr:.3f}")

            if pred == 1:
                st.error("High risk of sepsis (Positive flag)")
                st.caption("Use as an early warning signal; clinical confirmation is still required.")
            else:
                st.success("Lower risk of sepsis (Negative flag)")
                st.caption("Continue routine monitoring and reassessment as needed.")



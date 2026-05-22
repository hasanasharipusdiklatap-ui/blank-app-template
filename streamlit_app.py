from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dashboard Prediksi Realisasi 95%",
    page_icon="📊",
    layout="wide",
)

MODEL_PATH = Path("model/Best_model_skl.pkl")
DATA_PATH = Path("data/02_realisasi_anggaran_klasifikasi.csv")

TIPE_SATKER_OPTIONS = [
    "Dekonsentrasi",
    "Kantor Daerah",
    "Kantor Pusat",
    "Tugas Pembantuan",
]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def pagu_bucket_options() -> list[str]:
    return [
        "Semua",
        "< 10 miliar",
        "10 - 50 miliar",
        "50 - 100 miliar",
        "> 100 miliar",
    ]


def filter_data(
    data: pd.DataFrame,
    provinsi: str,
    pagu_bucket: str,
    jenis_belanja: str,
) -> pd.DataFrame:
    filtered = data.copy()
    if provinsi != "Semua":
        filtered = filtered[filtered["provinsi"] == provinsi]
    if jenis_belanja != "Semua":
        filtered = filtered[filtered["jenis_belanja_utama"] == jenis_belanja]
    if pagu_bucket != "Semua":
        if pagu_bucket == "< 10 miliar":
            filtered = filtered[filtered["pagu_miliar"] < 10]
        elif pagu_bucket == "10 - 50 miliar":
            filtered = filtered[(filtered["pagu_miliar"] >= 10) & (filtered["pagu_miliar"] <= 50)]
        elif pagu_bucket == "50 - 100 miliar":
            filtered = filtered[(filtered["pagu_miliar"] > 50) & (filtered["pagu_miliar"] <= 100)]
        else:
            filtered = filtered[filtered["pagu_miliar"] > 100]
    return filtered


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_feature_vector(
    tipe_satker: str,
    jumlah_spm: float,
    revisi_dipa: float,
    deviasi_rpd_persen: float,
    skor_ikpa: float,
) -> np.ndarray:
    one_hot = [1.0 if tipe_satker == choice else 0.0 for choice in TIPE_SATKER_OPTIONS]
    return np.array(
        [jumlah_spm, revisi_dipa, deviasi_rpd_persen, skor_ikpa, *one_hot],
        dtype=float,
    )


def predict(model, features: np.ndarray) -> tuple[str, float, float]:
    features = features.reshape(1, -1)
    proba = model.predict_proba(features)[0]
    pred = model.predict(features)[0]
    label = "Ya" if float(pred) == 1.0 else "Tidak"
    probability_yes = float(proba[1])
    probability_no = float(proba[0])
    return label, probability_yes, probability_no


def main():
    st.title("📊 Dashboard Prediksi Realisasi 95%")
    st.write(
        "Gunakan model terbaik untuk memprediksi apakah realisasi akan mencapai 95% dengan input anggaran dan karakteristik satker."
    )

    with st.sidebar:
        st.header("Input Prediksi")
        tipe_satker = st.selectbox("Tipe Satker", TIPE_SATKER_OPTIONS)
        jumlah_spm = st.number_input(
            "Jumlah SPM",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
        )
        revisi_dipa = st.number_input(
            "Revisi DIPA",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=1.0,
        )
        deviasi_rpd_persen = st.number_input(
            "Deviasi RPD (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1,
        )
        skor_ikpa = st.number_input(
            "Skor IKPA",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1,
        )
        st.markdown("---")
        st.caption("Model memanfaatkan fitur numerik dan encoding tipe satker.")

    data = load_data()
    model = load_model()

    with st.expander("Ringkasan Data", expanded=True):
        st.write("Dataset contoh prediksi dan statistik dasar dari data yang tersedia.")
        st.dataframe(data.head(10), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jumlah Baris", len(data))
            st.metric(
                "Persentase Ya",
                f"{(data['realisasi_tercapai_95persen'] == 'Ya').mean() * 100:.1f}%",
            )
        with col2:
            st.bar_chart(
                data['realisasi_tercapai_95persen'].value_counts().rename_axis('Label').reset_index(name='Count'),
                x='Label',
                y='Count',
            )

    st.markdown("---")
    st.subheader("Grafik Monitoring Realisasi")
    provinsi_options = ["Semua"] + sorted(data["provinsi"].dropna().unique().tolist())
    jenis_belanja_options = ["Semua"] + sorted(data["jenis_belanja_utama"].dropna().unique().tolist())
    selected_provinsi = st.selectbox("Filter Provinsi", provinsi_options)
    selected_pagu = st.selectbox("Filter Pagu", pagu_bucket_options())
    selected_jenis_belanja = st.selectbox("Filter Jenis Belanja", jenis_belanja_options)

    filtered_data = filter_data(data, selected_provinsi, selected_pagu, selected_jenis_belanja)
    if filtered_data.empty:
        st.warning("Tidak ada data yang sesuai dengan filter. Silakan ubah pilihan filter.")
    else:
        chart_data = filtered_data[
            ["realisasi_tw1_persen", "realisasi_tw2_persen", "realisasi_tw3_persen"]
        ].copy()
        chart_data = chart_data.rename(
            columns={
                "realisasi_tw1_persen": "TW1",
                "realisasi_tw2_persen": "TW2",
                "realisasi_tw3_persen": "TW3",
            }
        )
        chart_data = chart_data.melt(var_name="Triwulan", value_name="Realisasi (%)")
        summary = chart_data.groupby("Triwulan", as_index=False)["Realisasi (%)"].mean()

        line_chart = (
            alt.Chart(summary)
            .mark_line(point=True, color="#1f77b4", strokeWidth=3)
            .encode(
                x=alt.X("Triwulan:N", title="Triwulan"),
                y=alt.Y("Realisasi (%):Q", title="Rata-rata Realisasi (%)"),
                tooltip=["Triwulan", alt.Tooltip("Realisasi (%):Q", format=".2f")],
            )
            .properties(height=360)
        )
        st.altair_chart(line_chart, use_container_width=True)
        st.caption(
            f"Monitoring realisasi berdasarkan filter: Provinsi={selected_provinsi}, Pagu={selected_pagu}, Jenis Belanja={selected_jenis_belanja}."
        )

    st.subheader("Prediksi")
    if st.button("Hitung Prediksi"):
        features = build_feature_vector(
            tipe_satker,
            jumlah_spm,
            revisi_dipa,
            deviasi_rpd_persen,
            skor_ikpa,
        )
        predicted_label, prob_yes, prob_no = predict(model, features)

        st.success(f"Prediksi: **{predicted_label}**")
        st.write(
            f"Probabilitas mencapai 95% realisasi: **{prob_yes * 100:.2f}%**"
        )
        st.write(f"Probabilitas tidak mencapai 95% realisasi: **{prob_no * 100:.2f}%**")

        st.markdown("### Input yang digunakan")
        st.write(
            {
                "Tipe Satker": tipe_satker,
                "Jumlah SPM": jumlah_spm,
                "Revisi DIPA": revisi_dipa,
                "Deviasi RPD (%)": deviasi_rpd_persen,
                "Skor IKPA": skor_ikpa,
            }
        )

    st.markdown("---")
    st.caption("Model dimuat dari model/Best_model.pkcls dan dataset contoh dimuat dari data/02_realisasi_anggaran_klasifikasi.csv.")


if __name__ == "__main__":
    main()

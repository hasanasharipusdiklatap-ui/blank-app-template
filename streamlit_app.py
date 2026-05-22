from pathlib import Path

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


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


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

    data = load_data()
    model = load_model()

    provinsi_options = sorted(data['provinsi'].dropna().unique())
    jenis_options = sorted(data['jenis_belanja_utama'].dropna().unique())
    pagu_min = float(data['pagu_miliar'].min())
    pagu_max = float(data['pagu_miliar'].max())

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

    st.sidebar.header("Filter Monitoring")
    selected_provinsi = st.sidebar.multiselect(
        "Provinsi",
        options=provinsi_options,
        default=provinsi_options,
    )
    selected_jenis = st.sidebar.multiselect(
        "Jenis Belanja",
        options=jenis_options,
        default=jenis_options,
    )
    selected_pagu = st.sidebar.slider(
        "Rentang Pagu (miliar)",
        min_value=pagu_min,
        max_value=pagu_max,
        value=(pagu_min, pagu_max),
        step=0.1,
    )

    monitoring_data = data[
        data['provinsi'].isin(selected_provinsi)
        & data['jenis_belanja_utama'].isin(selected_jenis)
        & data['pagu_miliar'].between(selected_pagu[0], selected_pagu[1])
    ]

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
                data['realisasi_tercapai_95persen']
                .value_counts()
                .rename_axis('Label')
                .reset_index(name='Count'),
                x='Label',
                y='Count',
            )

    st.subheader("Monitoring Grafik")
    if monitoring_data.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
    else:
        st.markdown(
            "Filter monitoring digunakan untuk melihat kinerja sasaran realisasi berdasarkan provinsi, pagu, dan jenis belanja."
        )
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Baris yang Dipilih", len(monitoring_data))
        mcol2.metric(
            "Rata-rata Pagu",
            f"{monitoring_data['pagu_miliar'].mean():.2f} M",
        )
        mcol3.metric(
            "Rata-rata Skor IKPA",
            f"{monitoring_data['skor_ikpa'].mean():.2f}",
        )

        st.markdown("**Distribusi Realisasi 95%**")
        st.bar_chart(
            monitoring_data['realisasi_tercapai_95persen']
            .value_counts()
            .rename_axis('Label')
            .reset_index(name='Count')
            .set_index('Label')
        )

        st.markdown("**Jumlah Satker per Provinsi**")
        st.bar_chart(
            monitoring_data['provinsi']
            .value_counts()
            .rename_axis('Provinsi')
            .reset_index(name='Count')
            .set_index('Provinsi')
        )

        st.markdown("**Rata-rata Skor IKPA per Jenis Belanja**")
        st.bar_chart(
            monitoring_data.groupby('jenis_belanja_utama')['skor_ikpa']
            .mean()
            .sort_values(ascending=False)
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
    st.caption(
        "Model dimuat dari model/Best_model_skl.pkl dan dataset contoh dimuat dari data/02_realisasi_anggaran_klasifikasi.csv."
    )


if __name__ == "__main__":
    main()

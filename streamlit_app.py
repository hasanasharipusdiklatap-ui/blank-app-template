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
        "Gunakan tab di bawah untuk melihat overview data, memfilter grafik batang, atau menjalankan prediksi model."
    )

    with st.sidebar:
        st.header("Panduan")
        st.write(
            "Pilih tab untuk menavigasi: Overview Data, Grafik Monitoring, Prediksi Model."
        )
        st.markdown("---")
        st.caption("Data: provinsi, pagu, jenis belanja, realisasi, dan IKPA.")

    data = load_data()
    model = load_model()

    tab_overview, tab_grafik, tab_prediksi = st.tabs(
        ["Overview Data", "Grafik Monitoring", "Prediksi Model"]
    )

    with tab_overview:
        st.header("Overview Data Semua Provinsi")
        st.write(
            "Tampilan ringkas distribusi seluruh data per provinsi, realisasi, dan jenis belanja utama."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jumlah Baris", len(data))
            st.metric(
                "Persentase Ya",
                f"{(data['realisasi_tercapai_95persen'] == 'Ya').mean() * 100:.1f}%",
            )
            st.metric("Jumlah Provinsi", len(data["provinsi"].dropna().unique()))
        with col2:
            st.dataframe(data.head(10), use_container_width=True)
        st.markdown("### Distribusi Realisasi 95% per Provinsi")
        overview_chart = (
            alt.Chart(
                data.groupby(["provinsi", "realisasi_tercapai_95persen"]).size().reset_index(name="count")
            )
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("count:Q", title="Jumlah Satker"),
                y=alt.Y("provinsi:N", sort="-x", title="Provinsi"),
                color=alt.Color(
                    "realisasi_tercapai_95persen:N",
                    title="Realisasi 95%",
                    scale=alt.Scale(scheme="set2"),
                ),
                tooltip=[
                    alt.Tooltip("provinsi:N", title="Provinsi"),
                    alt.Tooltip("realisasi_tercapai_95persen:N", title="Realisasi 95%"),
                    alt.Tooltip("count:Q", title="Jumlah"),
                ],
            )
            .properties(height=520)
        )
        st.altair_chart(overview_chart, use_container_width=True)

    with tab_grafik:
        st.header("Grafik Batang Monitoring")
        st.write(
            "Pilih provinsi dan jenis belanja untuk melihat hasil filter pada grafik batang."
        )
        provinces = ["Semua"] + sorted(data["provinsi"].dropna().unique())
        selected_province = st.selectbox("Pilih Provinsi", provinces)
        jenis_belanja_options = ["Semua"] + sorted(data["jenis_belanja_utama"].dropna().unique())
        selected_jenis_belanja = st.selectbox(
            "Pilih Jenis Belanja Utama", jenis_belanja_options
        )
        min_pagu = float(data["pagu_miliar"].min())
        max_pagu = float(data["pagu_miliar"].max())
        selected_pagu = st.slider(
            "Rentang Pagu (miliar)",
            min_value=min_pagu,
            max_value=max_pagu,
            value=(min_pagu, max_pagu),
            step=0.1,
        )

        filtered = data.copy()
        if selected_province != "Semua":
            filtered = filtered[filtered["provinsi"] == selected_province]
        if selected_jenis_belanja != "Semua":
            filtered = filtered[
                filtered["jenis_belanja_utama"] == selected_jenis_belanja
            ]
        filtered = filtered[
            (filtered["pagu_miliar"] >= selected_pagu[0])
            & (filtered["pagu_miliar"] <= selected_pagu[1])
        ]

        st.write(
            f"Menampilkan {len(filtered)} baris data setelah filter: provinsi={selected_province}, pagu={selected_pagu[0]:.2f}-{selected_pagu[1]:.2f}, jenis belanja={selected_jenis_belanja}"
        )
        if filtered.empty:
            st.warning("Tidak ada data yang memenuhi filter saat ini.")
        else:
            chart_data = (
                filtered.groupby(["tipe_satker", "realisasi_tercapai_95persen"]).size().reset_index(name="count")
            )
            chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X("count:Q", title="Jumlah Satker"),
                    y=alt.Y("tipe_satker:N", sort="-x", title="Tipe Satker"),
                    color=alt.Color(
                        "realisasi_tercapai_95persen:N",
                        title="Realisasi 95%",
                        scale=alt.Scale(scheme="set2"),
                    ),
                    tooltip=[
                        alt.Tooltip("tipe_satker:N", title="Tipe Satker"),
                        alt.Tooltip("realisasi_tercapai_95persen:N", title="Realisasi 95%"),
                        alt.Tooltip("count:Q", title="Jumlah"),
                    ],
                )
                .properties(height=520)
            )
            st.altair_chart(chart, use_container_width=True)

    with tab_prediksi:
        st.header("Prediksi Model")
        st.write(
            "Masukkan nilai input yang diperlukan untuk memprediksi apakah realisasi akan mencapai 95%."
        )
        col1, col2 = st.columns([1, 1])
        with col1:
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
        with col2:
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
            st.write("\n")
            predict_button = st.button("Hitung Prediksi")

        if predict_button:
            features = build_feature_vector(
                tipe_satker,
                jumlah_spm,
                revisi_dipa,
                deviasi_rpd_persen,
                skor_ikpa,
            )
            predicted_label, prob_yes, prob_no = predict(model, features)
            st.success(f"Prediksi: **{predicted_label}**")
            st.write(f"Probabilitas mencapai 95% realisasi: **{prob_yes * 100:.2f}%**")
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
        else:
            st.info("Isi input lalu tekan tombol Hitung Prediksi.")

    st.markdown("---")
    st.caption(
        "Model dimuat dari model/Best_model_skl.pkl dan dataset contoh dimuat dari data/02_realisasi_anggaran_klasifikasi.csv."
    )


if __name__ == "__main__":
    main()

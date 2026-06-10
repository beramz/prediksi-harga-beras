import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Beras Jawa Tengah",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Prediksi Harga Beras Jawa Tengah 2025")
st.caption("Model: Random Forest Regression | Data: PIHPS Nasional")

# ─────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models_beras.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data_beras_bersih.csv")
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    return df

try:
    models = load_model()
    df     = load_data()
except FileNotFoundError:
    st.error("❌ File models_beras.pkl atau data_beras_bersih.csv tidak ditemukan. Pastikan kedua file ada di folder yang sama dengan app.py")
    st.stop()

# ─────────────────────────────────────────────
# KONFIGURASI LABEL & WARNA
# ─────────────────────────────────────────────
jenis_beras = ['bawah_1', 'bawah_2', 'medium_1', 'medium_2', 'super_1', 'super_2']
label_bersih = {
    'bawah_1':'Bawah I', 'bawah_2':'Bawah II',
    'medium_1':'Medium I', 'medium_2':'Medium II',
    'super_1':'Super I', 'super_2':'Super II',
}
warna = {
    'bawah_1':'#2196F3', 'bawah_2':'#64B5F6',
    'medium_1':'#FF9800', 'medium_2':'#FFB74D',
    'super_1':'#E91E63', 'super_2':'#F06292',
}

# ─────────────────────────────────────────────
# FEATURE ENGINEERING (sama persis dengan Colab)
# ─────────────────────────────────────────────
def buat_fitur(df, kolom_target):
    d = df[['tanggal', kolom_target]].copy()
    d = d.rename(columns={kolom_target: 'harga'})
    d['lag_1']           = d['harga'].shift(1)
    d['lag_3']           = d['harga'].shift(3)
    d['lag_7']           = d['harga'].shift(7)
    d['lag_14']          = d['harga'].shift(14)
    d['rolling_mean_7']  = d['harga'].shift(1).rolling(7).mean()
    d['rolling_mean_30'] = d['harga'].shift(1).rolling(30).mean()
    d['rolling_std_7']   = d['harga'].shift(1).rolling(7).std()
    d['bulan']           = d['tanggal'].dt.month
    d['hari_ke']         = d['tanggal'].dt.dayofyear
    d['minggu']          = d['tanggal'].dt.isocalendar().week.astype(int)
    d = d.dropna().reset_index(drop=True)
    fitur = ['lag_1','lag_3','lag_7','lag_14',
             'rolling_mean_7','rolling_mean_30','rolling_std_7',
             'bulan','hari_ke','minggu']
    return d, fitur

def split_data(d, rasio=0.80):
    n       = len(d)
    n_train = int(n * rasio)
    return d.iloc[:n_train], d.iloc[n_train:]

# ─────────────────────────────────────────────
# SIDEBAR — FILTER
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    jenis_dipilih = st.multiselect(
        "Pilih jenis beras",
        options=jenis_beras,
        default=jenis_beras,
        format_func=lambda x: label_bersih[x]
    )
    if not jenis_dipilih:
        st.warning("Pilih minimal 1 jenis beras")
        st.stop()

# ─────────────────────────────────────────────
# TAB NAVIGASI
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 Tren Harga Historis",
    "🎯 Aktual vs Prediksi",
    "📊 Evaluasi Model"
])

# ══════════════════════════════════════════════
# TAB 1: TREN HARGA HISTORIS
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Tren Harga Beras Harian — Jawa Tengah 2025")

    fig, ax = plt.subplots(figsize=(14, 5))
    for jenis in jenis_dipilih:
        ax.plot(df['tanggal'], df[jenis],
                label=label_bersih[jenis],
                color=warna[jenis], linewidth=1.5)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (Rp/kg)")
    ax.legend(ncol=3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Statistik ringkas
    st.subheader("Statistik Harga")
    stat_rows = []
    for jenis in jenis_dipilih:
        stat_rows.append({
            'Jenis Beras'  : label_bersih[jenis],
            'Harga Min (Rp)': f"Rp{df[jenis].min():,.0f}",
            'Harga Max (Rp)': f"Rp{df[jenis].max():,.0f}",
            'Rata-rata (Rp)': f"Rp{df[jenis].mean():,.0f}",
            'Std Dev (Rp)'  : f"Rp{df[jenis].std():,.0f}",
        })
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 2: AKTUAL VS PREDIKSI
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Aktual vs Prediksi Random Forest")

    n_col = 2 if len(jenis_dipilih) > 1 else 1
    cols  = st.columns(n_col)

    for i, jenis in enumerate(jenis_dipilih):
        d_jenis, fitur = buat_fitur(df, jenis)
        _, test        = split_data(d_jenis)

        model  = models[jenis]['model']
        y_pred = model.predict(test[fitur])
        y_test = test['harga'].values
        dates  = pd.to_datetime(test['tanggal'])

        # Filter hanya sampai Des 2025
        mask  = dates <= '2025-12-31'
        dates = dates[mask]
        y_test = y_test[mask]
        y_pred = y_pred[mask]

        with cols[i % n_col]:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(dates, y_test, label='Aktual',
                    color='#1565C0', linewidth=2)
            ax.plot(dates, y_pred, label='Prediksi',
                    color='#EF6C00', linewidth=1.8, linestyle='--')
            ax.set_title(
                f"{label_bersih[jenis]} | MAPE: {models[jenis]['MAPE']:.2f}%",
                fontsize=11, fontweight='bold'
            )
            ax.set_ylabel('Harga (Rp/kg)')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════
# TAB 3: EVALUASI MODEL
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Rekapitulasi Evaluasi Model")

    # Tabel evaluasi
    rekap_rows = []
    for jenis in jenis_dipilih:
        rekap_rows.append({
            'Jenis Beras' : label_bersih[jenis],
            'MAE (Rp)'    : f"Rp{models[jenis]['MAE']:,.2f}",
            'RMSE (Rp)'   : f"Rp{models[jenis]['RMSE']:,.2f}",
            'MAPE (%)'    : f"{models[jenis]['MAPE']:.2f}%",
        })
    df_rekap = pd.DataFrame(rekap_rows)
    st.dataframe(df_rekap, use_container_width=True, hide_index=True)

    # Rata-rata
    mape_vals = [models[j]['MAPE'] for j in jenis_dipilih]
    mae_vals  = [models[j]['MAE']  for j in jenis_dipilih]
    rmse_vals = [models[j]['RMSE'] for j in jenis_dipilih]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata MAE",  f"Rp{np.mean(mae_vals):,.2f}")
    col2.metric("Rata-rata RMSE", f"Rp{np.mean(rmse_vals):,.2f}")
    col3.metric("Rata-rata MAPE", f"{np.mean(mape_vals):.2f}%")

    st.divider()

    # Bar chart MAPE
    st.subheader("Perbandingan MAPE per Jenis Beras")
    fig, ax = plt.subplots(figsize=(10, 4))
    labels  = [label_bersih[j] for j in jenis_dipilih]
    values  = [models[j]['MAPE'] for j in jenis_dipilih]
    bars    = ax.bar(labels, values, color=[warna[j] for j in jenis_dipilih], width=0.5)
    ax.set_ylabel('MAPE (%)')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f'{v:.2f}%', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Bar chart MAE & RMSE
    st.subheader("Perbandingan MAE & RMSE per Jenis Beras")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, metric, color, title in zip(
        axes,
        [mae_vals, rmse_vals],
        ['#42A5F5', '#FFA726'],
        ['MAE (Rp)', 'RMSE (Rp)']
    ):
        bars = ax.bar(labels, metric, color=color, width=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Rp')
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, metric):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                    f'Rp{v:,.0f}', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
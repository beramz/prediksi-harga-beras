import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Beras Jawa Tengah",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.error("❌ File models_beras.pkl atau data_beras_bersih.csv tidak ditemukan.")
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
# DETEKSI & TRANSPOSE FORMAT PIHPS (HORIZONTAL)
# ─────────────────────────────────────────────
def deteksi_format_horizontal(df_mentah):
    """
    Mendeteksi apakah file berformat horizontal ala PIHPS
    (tanggal ada di kolom/header, jenis beras ada di baris).
    """
    kolom_wajib = ['tanggal', 'bawah_1', 'bawah_2',
                   'medium_1', 'medium_2', 'super_1', 'super_2']
    kolom_vertikal_ada = all(k in df_mentah.columns for k in kolom_wajib)
    if kolom_vertikal_ada:
        return False

    # Cek apakah baris pertama berisi pola nama beras PIHPS
    teks_gabungan = df_mentah.astype(str).apply(lambda x: ' '.join(x), axis=1).str.lower()
    pola_beras = ['bawah', 'medium', 'super', 'beras']
    cocok = teks_gabungan.str.contains('|'.join(pola_beras)).sum()
    return cocok >= 2


def transpose_format_pihps(df_mentah):
    """
    Mengubah data horizontal PIHPS (tanggal di header, jenis beras di baris)
    menjadi format vertikal (tanggal di baris, jenis beras di kolom).
    """
    df_raw = pd.read_excel(df_mentah, header=None, engine='openpyxl') \
        if hasattr(df_mentah, 'read') and df_mentah.name.endswith('.xlsx') \
        else df_mentah

    # Cari baris header tanggal — baris yang punya format dd/mm/yyyy terbanyak
    baris_tanggal_idx = None
    for i in range(min(5, len(df_raw))):
        baris = df_raw.iloc[i].astype(str)
        n_match = baris.str.match(r'^\d{1,2}/?\s?\d{1,2}/?\s?\d{2,4}$').sum()
        if n_match >= 3:
            baris_tanggal_idx = i
            break
    if baris_tanggal_idx is None:
        baris_tanggal_idx = 0

    # Cari kolom mulai data tanggal (lewati kolom No & Komoditas)
    kolom_mulai = 2
    for c in range(min(5, df_raw.shape[1])):
        val = str(df_raw.iloc[baris_tanggal_idx, c])
        if '/' in val or '-' in val:
            kolom_mulai = c
            break

    tanggal_raw = df_raw.iloc[baris_tanggal_idx, kolom_mulai:].tolist()
    tanggal = pd.to_datetime(tanggal_raw, dayfirst=True, errors='coerce')

    # Cari baris tiap jenis beras berdasarkan nama
    mapping_nama = {
        'bawah i'  : 'bawah_1', 'bawah ii' : 'bawah_2',
        'medium i' : 'medium_1', 'medium ii': 'medium_2',
        'super i'  : 'super_1', 'super ii' : 'super_2',
    }
    hasil = {'tanggal': tanggal}
    for i in range(len(df_raw)):
        nama_baris = str(df_raw.iloc[i, 1]).strip().lower()
        for key, kolom in mapping_nama.items():
            if key in nama_baris:
                nilai_raw = df_raw.iloc[i, kolom_mulai:].tolist()
                nilai_bersih = [str(v).replace(',', '').strip()
                                if v is not None else None for v in nilai_raw]
                hasil[kolom] = pd.to_numeric(nilai_bersih, errors='coerce')

    df_vertikal = pd.DataFrame(hasil)
    df_vertikal = df_vertikal.dropna(subset=['tanggal']).reset_index(drop=True)
    return df_vertikal


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
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
    n = len(d)
    n_train = int(n * rasio)
    return d.iloc[:n_train], d.iloc[n_train:]

def prediksi_ke_depan(df, jenis, model, fitur, n_hari=30):
    history   = df[jenis].values.tolist()
    last_date = df['tanggal'].iloc[-1]
    preds     = []
    for i in range(n_hari):
        s   = pd.Series(history)
        row = {
            'lag_1'           : s.iloc[-1],
            'lag_3'           : s.iloc[-3],
            'lag_7'           : s.iloc[-7],
            'lag_14'          : s.iloc[-14],
            'rolling_mean_7'  : s.iloc[-7:].mean(),
            'rolling_mean_30' : s.iloc[-30:].mean(),
            'rolling_std_7'   : s.iloc[-7:].std(),
            'bulan'           : (last_date + timedelta(days=i+1)).month,
            'hari_ke'         : (last_date + timedelta(days=i+1)).timetuple().tm_yday,
            'minggu'          : (last_date + timedelta(days=i+1)).isocalendar()[1],
        }
        pred = model.predict(pd.DataFrame([row])[fitur])[0]
        preds.append(pred)
        history.append(pred)
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_hari)
    return future_dates, preds

# ─────────────────────────────────────────────
# SIDEBAR NAVIGASI
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Rice_Paddy.jpg/320px-Rice_Paddy.jpg",
             use_column_width=True)
    st.markdown("## 🌾 Menu Navigasi")
    menu = st.radio("Pilih halaman:", [
        "🏠 Beranda",
        "📂 Upload Data Excel/CSV",
        "✏️ Input Data Manual",
        "📈 Tren Harga",
        "🎯 Aktual vs Prediksi",
        "📊 Evaluasi Model",
        "🔮 Prediksi ke Depan",
    ])
    st.divider()
    st.caption("Bramadita Rahardiyan Purnama\n220103196 | 2026")


# ══════════════════════════════════════════════
# 1. BERANDA / LANDING PAGE
# ══════════════════════════════════════════════
if menu == "🏠 Beranda":
    st.title("🌾 Sistem Prediksi Harga Beras Jawa Tengah")
    st.subheader("Berbasis Random Forest Regression")
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        Aplikasi ini merupakan implementasi model **Random Forest Regression**
        untuk memprediksi harga beras di Provinsi Jawa Tengah berdasarkan
        data harga harian dari **Pusat Informasi Harga Pangan Strategis (PIHPS)**
        periode Januari hingga Desember 2025.

        **Jenis beras yang diprediksi:**
        - Beras Kualitas Bawah I & II
        - Beras Kualitas Medium I & II
        - Beras Kualitas Super I & II
        """)

    with col2:
        st.metric("Total Data", "365 hari")
        st.metric("Jenis Beras", "6 jenis")
        st.metric("Rata-rata MAPE", "0.93%")

    st.divider()
    st.markdown("### 📋 Panduan Penggunaan")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**📂 Upload Data**\nUpload file Excel/CSV untuk prediksi batch sekaligus")
    with col2:
        st.info("**✏️ Input Manual**\nPilih jenis beras dan tanggal untuk prediksi satu per satu")
    with col3:
        st.info("**📈 Tren Harga**\nLihat grafik pergerakan harga beras sepanjang 2025")
    with col4:
        st.info("**🔮 Prediksi**\nLihat prediksi harga beras hingga 30 hari ke depan")

    st.divider()
    st.markdown("### 📊 Ringkasan Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Periode Data", "Jan – Des 2025")
    with col2:
        st.metric("Sumber Data", "PIHPS Nasional")
    with col3:
        st.metric("Algoritma", "Random Forest Regression")


# ══════════════════════════════════════════════
# 2. UPLOAD DATA EXCEL/CSV
# ══════════════════════════════════════════════
elif menu == "📂 Upload Data Excel/CSV":
    st.title("📂 Upload Data untuk Prediksi")
    st.caption("Upload file Excel atau CSV untuk mendapatkan hasil prediksi.")

    # Template download
    st.markdown("#### Format file yang diperlukan:")
    df_contoh = pd.DataFrame({
        'tanggal'  : ['2025-01-01', '2025-01-02', '2025-01-03'],
        'bawah_1'  : [13200, 13250, 13200],
        'bawah_2'  : [13000, 12350, 13000],
        'medium_1' : [15050, 15100, 15050],
        'medium_2' : [14150, 13950, 14150],
        'super_1'  : [18050, 18250, 18050],
        'super_2'  : [16450, 14500, 16450],
    })
    st.dataframe(df_contoh, use_container_width=True, hide_index=True)
    csv_template = df_contoh.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Template CSV", csv_template,
                       "template_data_beras.csv", "text/csv")

    st.divider()
    uploaded_file = st.file_uploader("Upload file di sini", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            # ── TAHAP 0: Baca file & deteksi format ──
            if uploaded_file.name.endswith('.csv'):
                df_cek = pd.read_csv(uploaded_file, header=None)
            else:
                df_cek = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

            format_horizontal = deteksi_format_horizontal(
                pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv')
                else pd.read_excel(uploaded_file)
            )

            if format_horizontal:
                st.info("🔄 Terdeteksi format data horizontal (gaya PIHPS — tanggal di kolom, "
                        "jenis beras di baris). Data akan ditransformasi ke format vertikal secara otomatis.")
                uploaded_file.seek(0)
                if uploaded_file.name.endswith('.xlsx'):
                    df_mentah = transpose_format_pihps(uploaded_file)
                else:
                    # Untuk CSV horizontal, baca dulu sebagai raw lalu transpose manual
                    df_raw_csv = pd.read_csv(uploaded_file, header=None)
                    tmp_path = "_tmp_upload.xlsx"
                    df_raw_csv.to_excel(tmp_path, header=False, index=False)
                    with open(tmp_path, 'rb') as f:
                        df_mentah = transpose_format_pihps(f)

                st.success(f"✅ Berhasil ditransformasi ke format vertikal: {len(df_mentah)} baris, "
                           f"{len(df_mentah.columns)} kolom")
            else:
                uploaded_file.seek(0)
                if uploaded_file.name.endswith('.csv'):
                    df_mentah = pd.read_csv(uploaded_file)
                else:
                    df_mentah = pd.read_excel(uploaded_file)

            # ── TAHAP 1: Preview data ──
            st.markdown("### 📋 Tahap 1 — Preview Data")
            st.caption(f"Total baris: {len(df_mentah)} | Total kolom: {len(df_mentah.columns)}")
            st.dataframe(df_mentah.head(10), use_container_width=True, hide_index=True)

            # Cek masalah awal
            masalah = []
            kolom_wajib = ['tanggal', 'bawah_1', 'bawah_2',
                           'medium_1', 'medium_2', 'super_1', 'super_2']
            kolom_hilang = [k for k in kolom_wajib if k not in df_mentah.columns]
            if kolom_hilang:
                masalah.append(f"Kolom tidak ditemukan: {kolom_hilang}")
            if df_mentah.isnull().sum().sum() > 0:
                masalah.append(f"Ditemukan {df_mentah.isnull().sum().sum()} nilai kosong")
            for jenis in jenis_beras:
                if jenis in df_mentah.columns:
                    n_koma = df_mentah[jenis].astype(str).str.contains(',').sum()
                    if n_koma > 0:
                        masalah.append(f"Kolom {jenis}: {n_koma} baris berformat teks dengan koma")

            if masalah:
                st.warning("⚠️ Ditemukan masalah pada data mentah:\n" +
                           "\n".join([f"- {m}" for m in masalah]))
            else:
                st.success("✅ Tidak ditemukan masalah pada data mentah")

            st.divider()

            # ── TAHAP 2: Proses pembersihan & penataan ulang ──
            st.markdown("### 🧹 Tahap 2 — Pembersihan & Penataan Ulang Data")

            if kolom_hilang:
                st.error(f"❌ Kolom wajib tidak ditemukan: {kolom_hilang}. Periksa kembali file Anda.")
                st.stop()

            df_bersih  = df_mentah.copy()
            log_bersih = []

            # 1. Konversi & validasi tanggal
            df_bersih['tanggal'] = pd.to_datetime(df_bersih['tanggal'], errors='coerce')
            n_tgl_error = df_bersih['tanggal'].isna().sum()
            if n_tgl_error > 0:
                df_bersih = df_bersih.dropna(subset=['tanggal'])
                log_bersih.append(f"Hapus {n_tgl_error} baris dengan format tanggal tidak valid")
            else:
                log_bersih.append("Format tanggal valid, tidak ada yang dihapus")

            # 2. Hapus duplikat tanggal
            n_dup = df_bersih.duplicated(subset=['tanggal']).sum()
            if n_dup > 0:
                df_bersih = df_bersih.drop_duplicates(subset=['tanggal'], keep='first')
                log_bersih.append(f"Hapus {n_dup} baris duplikat tanggal")
            else:
                log_bersih.append("Tidak ada duplikat tanggal")

            # 3. Bersihkan harga: hapus koma & konversi ke numerik
            for jenis in jenis_beras:
                df_bersih[jenis] = (df_bersih[jenis].astype(str)
                                     .str.replace(',', '', regex=False)
                                     .str.replace('Rp', '', regex=False)
                                     .str.strip())
                df_bersih[jenis] = pd.to_numeric(df_bersih[jenis], errors='coerce')
            log_bersih.append("Koma/simbol Rp dihapus dari kolom harga, dikonversi ke numerik")

            # 4. Urutkan berdasarkan tanggal (penataan ulang)
            df_bersih = df_bersih.sort_values('tanggal').reset_index(drop=True)
            log_bersih.append("Data ditata ulang dan diurutkan berdasarkan tanggal")

            # 5. Resampling harian — lengkapi tanggal yang hilang
            tgl_awal = df_bersih['tanggal'].min()
            tgl_akhir = df_bersih['tanggal'].max()
            n_hari_seharusnya = (tgl_akhir - tgl_awal).days + 1
            if len(df_bersih) < n_hari_seharusnya:
                n_hilang = n_hari_seharusnya - len(df_bersih)
                df_bersih = df_bersih.set_index('tanggal')
                df_bersih = df_bersih.reindex(
                    pd.date_range(tgl_awal, tgl_akhir, freq='D')
                )
                df_bersih.index.name = 'tanggal'
                df_bersih = df_bersih.reset_index()
                log_bersih.append(f"Tambahkan {n_hilang} baris tanggal yang hilang (resampling harian)")
            else:
                log_bersih.append("Tidak ada tanggal yang hilang, resampling tidak diperlukan")

            # 6. Tangani missing value dengan interpolasi linear
            n_missing = df_bersih[jenis_beras].isnull().sum().sum()
            if n_missing > 0:
                df_bersih[jenis_beras] = df_bersih[jenis_beras].interpolate(method='linear')
                log_bersih.append(f"{n_missing} nilai kosong diisi dengan interpolasi linear")
            else:
                log_bersih.append("Tidak ada nilai kosong setelah resampling")

            for log in log_bersih:
                st.markdown(f"- ✅ {log}")

            st.divider()

            # ── TAHAP 3: Preview data bersih ──
            st.markdown("### ✅ Tahap 3 — Preview Data Setelah Dibersihkan & Ditata Ulang")
            st.caption(f"Total baris valid: {len(df_bersih)}")
            st.dataframe(df_bersih.head(10), use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Baris sebelum", len(df_mentah))
            col2.metric("Baris sesudah", len(df_bersih))
            col3.metric("Selisih baris",
                        len(df_bersih) - len(df_mentah))

            csv_bersih = df_bersih.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Data yang Sudah Dibersihkan", csv_bersih,
                               "data_bersih_hasil_upload.csv", "text/csv")

            st.divider()

            # ── TAHAP 4: Prediksi ──
            st.markdown("### 🔍 Tahap 4 — Hasil Prediksi")

            df_gabung = pd.concat([df, df_bersih]).drop_duplicates('tanggal').sort_values('tanggal').reset_index(drop=True)

            jenis_pilih = st.multiselect("Pilih jenis beras:",
                                          jenis_beras,
                                          default=jenis_beras,
                                          format_func=lambda x: label_bersih[x])

            if jenis_pilih:
                n_col = 2 if len(jenis_pilih) > 1 else 1
                cols  = st.columns(n_col)

                for i, jenis in enumerate(jenis_pilih):
                    d_jenis, fitur = buat_fitur(df_gabung, jenis)
                    mask_upload    = d_jenis['tanggal'].isin(df_bersih['tanggal'])
                    d_uji          = d_jenis[mask_upload]

                    if len(d_uji) == 0:
                        st.warning(f"{label_bersih[jenis]}: tidak ada data yang bisa diprediksi "
                                   f"(kemungkinan tanggal terlalu dekat dengan awal data sehingga "
                                   f"fitur lag/rolling belum dapat dihitung)")
                        continue

                    model  = models[jenis]['model']
                    y_pred = model.predict(d_uji[fitur])
                    y_act  = d_uji['harga'].values
                    mae    = np.mean(np.abs(y_act - y_pred))
                    mape   = np.mean(np.abs((y_act - y_pred) / y_act)) * 100

                    with cols[i % n_col]:
                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        ax.plot(d_uji['tanggal'], y_act,
                                label='Aktual', color='#1565C0', linewidth=2)
                        ax.plot(d_uji['tanggal'], y_pred,
                                label='Prediksi', color='#EF6C00',
                                linewidth=1.8, linestyle='--')
                        ax.set_title(
                            f"{label_bersih[jenis]} | MAE: Rp{mae:,.0f} | MAPE: {mape:.2f}%",
                            fontsize=10, fontweight='bold')
                        ax.set_ylabel('Harga (Rp/kg)')
                        ax.legend(fontsize=8)
                        ax.grid(axis='y', alpha=0.3)
                        ax.tick_params(axis='x', rotation=30)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

        except Exception as e:
            st.error(f"❌ Error memproses file: {e}")


# ══════════════════════════════════════════════
# 3. INPUT DATA MANUAL
# ══════════════════════════════════════════════
elif menu == "✏️ Input Data Manual":
    st.title("✏️ Input Data Manual")
    st.caption("Pilih jenis beras dan tanggal untuk mendapatkan prediksi harga.")

    col1, col2 = st.columns(2)
    with col1:
        jenis_pilih = st.selectbox("Pilih jenis beras:",
                                    jenis_beras,
                                    format_func=lambda x: label_bersih[x])
    with col2:
        tgl_min = df['tanggal'].iloc[-1] + timedelta(days=1)
        tgl_max = df['tanggal'].iloc[-1] + timedelta(days=30)
        tanggal_input = st.date_input("Pilih tanggal prediksi:",
                                       value=tgl_min.date(),
                                       min_value=tgl_min.date(),
                                       max_value=tgl_max.date())

    if st.button("🔍 Prediksi Harga", type="primary"):
        # Hitung berapa hari ke depan dari data terakhir
        last_date  = df['tanggal'].iloc[-1]
        target_date = pd.Timestamp(tanggal_input)
        n_hari     = (target_date - last_date).days

        if n_hari <= 0:
            st.error("Tanggal yang dipilih harus setelah tanggal data terakhir.")
        else:
            model  = models[jenis_pilih]['model']
            fitur  = models[jenis_pilih]['fitur']

            future_dates, future_preds = prediksi_ke_depan(
                df, jenis_pilih, model, fitur, n_hari=n_hari
            )

            harga_prediksi = future_preds[-1]

            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Jenis Beras", label_bersih[jenis_pilih])
            col2.metric("Tanggal Prediksi", target_date.strftime('%d %B %Y'))
            col3.metric("Harga Prediksi", f"Rp{harga_prediksi:,.0f}/kg")

            # Grafik tren + prediksi
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df['tanggal'].iloc[-60:], df[jenis_pilih].iloc[-60:],
                    label='Historis', color='#1565C0', linewidth=2)
            ax.plot(future_dates, future_preds,
                    label='Prediksi', color='#EF6C00',
                    linewidth=1.8, linestyle='--', marker='o', markersize=4)
            ax.axvline(last_date, color='gray', linestyle=':', linewidth=1.5)
            ax.scatter([target_date], [harga_prediksi],
                       color='red', s=100, zorder=5,
                       label=f'Target: Rp{harga_prediksi:,.0f}')
            ax.set_title(f'Prediksi Harga {label_bersih[jenis_pilih]} — {target_date.strftime("%d %b %Y")}',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga (Rp/kg)')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════
# 4. TREN HARGA
# ══════════════════════════════════════════════
elif menu == "📈 Tren Harga":
    st.title("📈 Tren Harga Beras Harian")
    st.caption("Jawa Tengah, Januari – Desember 2025 | Sumber: PIHPS Nasional")

    jenis_pilih = st.multiselect("Pilih jenis beras:",
                                  jenis_beras,
                                  default=jenis_beras,
                                  format_func=lambda x: label_bersih[x])

    if jenis_pilih:
        fig, ax = plt.subplots(figsize=(16, 5))
        for jenis in jenis_pilih:
            ax.plot(df['tanggal'], df[jenis],
                    label=label_bersih[jenis],
                    color=warna[jenis], linewidth=1.5)
        ax.set_title('Tren Harga Beras Harian — Jawa Tengah 2025',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga (Rp/kg)')
        ax.legend(ncol=3)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Statistik Deskriptif")
        stat_rows = []
        for jenis in jenis_pilih:
            stat_rows.append({
                'Jenis Beras'   : label_bersih[jenis],
                'Harga Min'     : f"Rp{df[jenis].min():,.0f}",
                'Harga Max'     : f"Rp{df[jenis].max():,.0f}",
                'Rata-rata'     : f"Rp{df[jenis].mean():,.0f}",
                'Std Deviasi'   : f"Rp{df[jenis].std():,.0f}",
            })
        st.dataframe(pd.DataFrame(stat_rows),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# 5. AKTUAL VS PREDIKSI
# ══════════════════════════════════════════════
elif menu == "🎯 Aktual vs Prediksi":
    st.title("🎯 Aktual vs Prediksi Random Forest")
    st.caption("Perbandingan harga aktual dan hasil prediksi pada data uji (20%) periode Nov–Des 2025")

    jenis_pilih = st.multiselect("Pilih jenis beras:",
                                  jenis_beras,
                                  default=jenis_beras,
                                  format_func=lambda x: label_bersih[x])

    if jenis_pilih:
        n_col = 2 if len(jenis_pilih) > 1 else 1
        cols  = st.columns(n_col)

        for i, jenis in enumerate(jenis_pilih):
            d_jenis, fitur = buat_fitur(df, jenis)
            _, test        = split_data(d_jenis)
            model          = models[jenis]['model']
            y_pred         = model.predict(test[fitur])
            y_test         = test['harga'].values
            dates          = pd.to_datetime(test['tanggal'])
            mask           = dates <= '2025-12-31'

            with cols[i % n_col]:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.plot(dates[mask], y_test[mask],
                        label='Aktual', color='#1565C0', linewidth=2)
                ax.plot(dates[mask], y_pred[mask],
                        label='Prediksi', color='#EF6C00',
                        linewidth=1.8, linestyle='--')
                ax.set_title(
                    f"{label_bersih[jenis]}  |  MAPE: {models[jenis]['MAPE']:.2f}%",
                    fontsize=11, fontweight='bold')
                ax.set_ylabel('Harga (Rp/kg)')
                ax.legend(fontsize=8)
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


# ══════════════════════════════════════════════
# 6. EVALUASI MODEL
# ══════════════════════════════════════════════
elif menu == "📊 Evaluasi Model":
    st.title("📊 Evaluasi Model Random Forest Regression")

    # Tabel evaluasi
    rekap_rows = []
    for jenis in jenis_beras:
        rekap_rows.append({
            'Jenis Beras' : label_bersih[jenis],
            'MAE (Rp)'    : f"Rp{models[jenis]['MAE']:,.2f}",
            'RMSE (Rp)'   : f"Rp{models[jenis]['RMSE']:,.2f}",
            'MAPE (%)'    : f"{models[jenis]['MAPE']:.2f}%",
        })
    st.subheader("Rekapitulasi Evaluasi")
    st.dataframe(pd.DataFrame(rekap_rows),
                 use_container_width=True, hide_index=True)

    # Kartu metrik rata-rata
    st.divider()
    mape_vals = [models[j]['MAPE'] for j in jenis_beras]
    mae_vals  = [models[j]['MAE']  for j in jenis_beras]
    rmse_vals = [models[j]['RMSE'] for j in jenis_beras]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata MAE",  f"Rp{np.mean(mae_vals):,.2f}")
    col2.metric("Rata-rata RMSE", f"Rp{np.mean(rmse_vals):,.2f}")
    col3.metric("Rata-rata MAPE", f"{np.mean(mape_vals):.2f}%")

    st.divider()

    # Bar chart MAE, RMSE, MAPE
    labels = [label_bersih[j] for j in jenis_beras]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, vals, color, title in zip(
        axes,
        [mae_vals, rmse_vals, mape_vals],
        ['#42A5F5', '#FFA726', '#EC407A'],
        ['MAE (Rp)', 'RMSE (Rp)', 'MAPE (%)']
    ):
        bars = ax.bar(labels, vals, color=color, width=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Rp' if 'Rp' in title else '%')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
        for bar, v in zip(bars, vals):
            label_val = f'Rp{v:,.0f}' if 'Rp' in title else f'{v:.2f}%'
            ax.text(bar.get_x() + bar.get_width()/2, v + (max(vals)*0.01),
                    label_val, ha='center', fontsize=9)

    plt.suptitle('Perbandingan Evaluasi Model — 6 Jenis Beras',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
# 7. PREDIKSI KE DEPAN
# ══════════════════════════════════════════════
elif menu == "🔮 Prediksi ke Depan":
    st.title("🔮 Prediksi Harga Beras ke Depan")
    st.caption(f"Prediksi dihitung mulai {(df['tanggal'].iloc[-1] + timedelta(days=1)).strftime('%d %B %Y')}")

    col1, col2 = st.columns(2)
    with col1:
        jenis_pilih = st.selectbox("Pilih jenis beras:",
                                    jenis_beras,
                                    format_func=lambda x: label_bersih[x])
    with col2:
        n_hari = st.slider("Jumlah hari prediksi:", min_value=7,
                           max_value=30, value=30, step=1)

    if st.button("🔮 Jalankan Prediksi", type="primary"):
        model        = models[jenis_pilih]['model']
        fitur        = models[jenis_pilih]['fitur']
        future_dates, future_preds = prediksi_ke_depan(
            df, jenis_pilih, model, fitur, n_hari=n_hari
        )

        # Grafik
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df['tanggal'].iloc[-60:], df[jenis_pilih].iloc[-60:],
                label='Historis', color='#1565C0', linewidth=2)
        ax.plot(future_dates, future_preds,
                label=f'Prediksi {n_hari} hari ke depan',
                color='#EF6C00', linewidth=2,
                linestyle='--', marker='o', markersize=4)
        ax.axvline(df['tanggal'].iloc[-1], color='gray',
                   linestyle=':', linewidth=1.5, label='Akhir data')
        ax.set_title(
            f'Prediksi {n_hari} Hari ke Depan — {label_bersih[jenis_pilih]}',
            fontsize=13, fontweight='bold')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga (Rp/kg)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Tabel prediksi
        st.divider()
        st.subheader("Tabel Hasil Prediksi")
        df_pred = pd.DataFrame({
            'Tanggal'        : [d.strftime('%d %B %Y') for d in future_dates],
            'Prediksi Harga' : [f"Rp{p:,.0f}" for p in future_preds],
        })
        st.dataframe(df_pred, use_container_width=True, hide_index=True)

        # Download
        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            csv,
            f"prediksi_{jenis_pilih}_{n_hari}hari.csv",
            "text/csv"
        )

        # Ringkasan
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Harga Terakhir (historis)",
                    f"Rp{df[jenis_pilih].iloc[-1]:,.0f}")
        col2.metric("Prediksi Hari Pertama",
                    f"Rp{future_preds[0]:,.0f}")
        col3.metric(f"Prediksi Hari ke-{n_hari}",
                    f"Rp{future_preds[-1]:,.0f}")

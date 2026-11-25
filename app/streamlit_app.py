import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="AML Transaction Monitoring & MRR Risk Analytics",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kri = pd.read_parquet(os.path.join(base_dir, 'data', 'processed', 'bank_kri_summary.parquet'))
    alerts = pd.read_parquet(os.path.join(base_dir, 'data', 'processed', 'suspicious_alerts.parquet'))
    return kri, alerts

kri_df, alerts_df = load_data()

# ================= SIDEBAR: FILTER & SIMULASI THRESHOLD =================
st.sidebar.header("🎛️ Filter Pemantauan MRR & AML")

kri_statuses = sorted(kri_df['KRI_Status'].dropna().unique().tolist())
selected_kri = st.sidebar.multiselect(
    "Status KRI Unit / Bank:",
    options=kri_statuses,
    default=kri_statuses
)

min_score = st.sidebar.slider(
    "Ambang Batas AML Risk Score (STR Threshold):",
    min_value=65,
    max_value=100,
    value=75,
    step=1
)

payment_formats = sorted(alerts_df['Payment_Format_Label'].dropna().unique().tolist())
selected_formats = st.sidebar.multiselect(
    "Kanal / Format Pembayaran:",
    options=payment_formats,
    default=payment_formats
)

# Terapkan Filter
kri_filt = kri_df[kri_df['KRI_Status'].isin(selected_kri)]
alerts_filt = alerts_df[
    (alerts_df['From_Bank'].isin(kri_filt['From_Bank'])) &
    (alerts_df['AML_Risk_Score'] >= min_score) &
    (alerts_df['Payment_Format_Label'].isin(selected_formats))
]

# ================= HEADER & EXECUTIVE METRICS =================
st.title("🛡️ AML Transaction Monitoring & Regional Risk (MRR) Platform")
st.markdown(
    "Sistem Pemantauan **Key Risk Indicators (KRI)** Unit Bank, Deteksi Tipologi Jaringan (*Fan-In / Fan-Out*), "
    "dan Antrean Investigasi **LTKM / STR** Berbasis *Hybrid Rule-Engine & Tuned LightGBM*."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Institusi / Unit Dipantau", f"{len(kri_filt):,} Unit")
c2.metric("Unit Status RED (Immediate Audit)", f"{(kri_filt['KRI_Status'] == 'Red (Immediate Audit)').sum():,} Unit")
c3.metric("High-Risk STR Alerts (Aktif)", f"{len(alerts_filt):,} Transaksi")
c4.metric("Total Eksposur Dana Mencurigakan", f"${alerts_filt['Amount_Paid'].sum():,.0f}")

st.markdown("---")

# ================= TABS ANALISIS =================
tab1, tab2, tab3 = st.tabs([
    "📊 KRI Unit Bank & Status RAG",
    "🕸️ Analisis Tipologi & Kanal Transaksi",
    "🔍 Antrean Investigasi STR / LTKM & Export"
])

with tab1:
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.subheader("Distribusi Status KRI Unit Bank")
        fig_pie = px.pie(
            kri_filt,
            names='KRI_Status',
            hole=0.45,
            color='KRI_Status',
            color_discrete_map={
                'Green (Compliant)': '#2ca02c',
                'Amber (Enhanced Monitoring)': '#ff7f0e',
                'Red (Immediate Audit)': '#d62728'
            }
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Top 15 Unit Bank dengan Rasio Red Alert Tertinggi")
        top_banks = kri_filt.sort_values(by=['Red_Alert_Rate', 'Red_Alerts'], ascending=False).head(15)
        top_banks['From_Bank_Label'] = "Bank " + top_banks['From_Bank'].astype(str)
        fig_bar = px.bar(
            top_banks,
            x='From_Bank_Label',
            y='Red_Alert_Rate',
            color='KRI_Status',
            color_discrete_map={
                'Green (Compliant)': '#2ca02c',
                'Amber (Enhanced Monitoring)': '#ff7f0e',
                'Red (Immediate Audit)': '#d62728'
            },
            labels={'From_Bank_Label': 'Kode Bank / Unit', 'Red_Alert_Rate': 'Rasio Red Alert (%)'}
        )
        fig_bar.update_yaxes(tickformat=".2%")
        fig_bar.update_layout(margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Tabel Rekapitulasi KRI Tingkat Unit Bank")
    st.dataframe(
        kri_filt.sort_values(by=['Red_Alert_Rate', 'Avg_Risk_Score'], ascending=False),
        use_container_width=True
    )

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi Pemicu Red-Flag & Anomali")
        flag_counts = alerts_filt['Triggered_Red_Flags'].value_counts().head(10).reset_index()
        flag_counts.columns = ['Triggered_Red_Flags', 'Jumlah_Kasus']
        fig_flags = px.bar(
            flag_counts,
            x='Jumlah_Kasus',
            y='Triggered_Red_Flags',
            orientation='h',
            color='Jumlah_Kasus',
            color_continuous_scale='Reds',
            labels={'Jumlah_Kasus': 'Jumlah Transaksi', 'Triggered_Red_Flags': 'Indikator Red-Flag'}
        )
        fig_flags.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(t=20, b=20))
        st.plotly_chart(fig_flags, use_container_width=True)

    with col_b:
        st.subheader("Eksposur Dana Mencurigakan per Kanal Pembayaran")
        fmt_summary = alerts_filt.groupby('Payment_Format_Label', as_index=False)['Amount_Paid'].sum()
        fig_fmt = px.bar(
            fmt_summary.sort_values('Amount_Paid', ascending=False),
            x='Payment_Format_Label',
            y='Amount_Paid',
            color='Payment_Format_Label',
            labels={'Payment_Format_Label': 'Kanal Pembayaran', 'Amount_Paid': 'Total Nilai Transaksi ($)'}
        )
        fig_fmt.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_fmt, use_container_width=True)

    # Deteksi otomatis kolom yang tersedia untuk scatter plot
    has_graph_cols = {'Sender_Unique_Receivers', 'Receiver_Unique_Senders'}.issubset(alerts_filt.columns)
    sample_scatter = alerts_filt.head(2000)

    if has_graph_cols:
        st.subheader("Pemetaan Jaringan: Fan-In (Pengirim Unik) vs Fan-Out (Penerima Unik)")
        x_col, y_col = 'Sender_Unique_Receivers', 'Receiver_Unique_Senders'
        labels_map = {
            'Sender_Unique_Receivers': 'Fan-Out Degree (Jumlah Penerima Unik)',
            'Receiver_Unique_Senders': 'Fan-In Degree (Jumlah Pengirim Unik)',
            'AML_Risk_Score': 'AML Risk Score'
        }
    else:
        st.subheader("Pemetaan Perilaku: Kecepatan Transaksi (Velocity) vs Lonjakan Nominal")
        x_col, y_col = 'Sender_Hourly_Tx_Count', 'Amount_to_Sender_Avg_Ratio'
        labels_map = {
            'Sender_Hourly_Tx_Count': 'Frekuensi Transaksi per Jam (Velocity)',
            'Amount_to_Sender_Avg_Ratio': 'Rasio Nominal vs Rata-rata Rekening',
            'AML_Risk_Score': 'AML Risk Score'
        }

    fig_scatter = px.scatter(
        sample_scatter,
        x=x_col,
        y=y_col,
        size='Amount_Paid',
        color='AML_Risk_Score',
        color_continuous_scale='YlOrRd',
        hover_data=['From_Account', 'To_Account', 'Amount_Paid', 'Triggered_Red_Flags'],
        labels=labels_map
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Antrean Investigasi Kasus Berisiko Tinggi (Kertas Kerja STR / LTKM)")
    st.caption("Diurutkan berdasarkan skor risiko tertinggi (kombinasi probabilitas Tuned LightGBM + penalti kepatuhan APU-PPT).")

    base_cols = [
        'Timestamp', 'From_Bank', 'From_Account', 'From_Entity_Name',
        'To_Bank', 'To_Account', 'To_Entity_Name',
        'Amount_Paid', 'Payment_Currency_Label', 'Payment_Format_Label',
        'Sender_Hourly_Tx_Count', 'AML_Risk_Score', 'Alert_Tier', 'Triggered_Red_Flags'
    ]
    extra_graph = [c for c in ['Sender_Unique_Receivers', 'Receiver_Unique_Senders'] if c in alerts_filt.columns]
    display_cols = base_cols[:11] + extra_graph + base_cols[11:]

    st.dataframe(alerts_filt[display_cols].head(200), use_container_width=True)

    csv_data = alerts_filt[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Kertas Kerja Investigasi STR / LTKM (CSV)",
        data=csv_data,
        file_name="kertas_kerja_investigasi_aml_str.csv",
        mime="text/csv"
    )
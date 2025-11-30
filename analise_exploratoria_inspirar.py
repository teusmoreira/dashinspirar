import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURAÇÃO VISUAL ---
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),
    "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "font.size": 10
})

st.set_page_config(page_title="Dashboard Inspirar", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Dashboard de Engajamento - App Inspirar")
st.markdown("---") 

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📂 Configurações")
    uploaded_file = st.file_uploader("Carregar JSON", type=["json"])

# Caminho do arquivo
LOCAL_PATH = "pacientes_marco-julho_com_createdAt_com_sexo_sigla_filtrado.json"

@st.cache_data
def load_data(file_input):
    try:
        if isinstance(file_input, str):
            with open(file_input, "r", encoding="utf-8") as dataset:
                data = json.load(dataset)
        else:
            data = json.load(file_input)

        pacientes = pd.json_normalize(data["data"]["result"])
        
        # Tratamentos
        pacientes["createdAt"] = pd.to_datetime(pacientes["createdAt"], errors="coerce")
        pacientes["height"] = pd.to_numeric(pacientes["height"], errors='coerce')
        pacientes["height"] = np.where(pacientes["height"] > 3, pacientes["height"] / 100, pacientes["height"])
        
        # Scores
        pacientes["n_symptoms"] = pacientes["symptomDiaries"].apply(len)
        pacientes["n_acqs"] = pacientes["acqs"].apply(len)
        pacientes["n_prescriptions"] = pacientes["prescriptions"].apply(len)
        pacientes["n_activity_logs"] = pacientes["activityLogs"].apply(len)
        pacientes["engagement_score"] = (pacientes["n_symptoms"] + pacientes["n_acqs"] + 
                                         pacientes["n_prescriptions"] + pacientes["n_activity_logs"])
        
        pacientes["bmi"] = pacientes["weight"] / (pacientes["height"] ** 2)
        return pacientes
    except Exception as e:
        return None

# --- CARREGAMENTO ---
raw_df = None
if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
elif raw_df is None:
    try:
        raw_df = load_data(LOCAL_PATH)
    except:
        pass

# --- FILTROS E VISUALIZAÇÃO ---
if raw_df is not None:
    
    # 1. Detectar datas
    min_date = raw_df["createdAt"].min().date()
    max_date = raw_df["createdAt"].max().date()
    
    # 2. Filtro na Sidebar
    st.sidebar.divider()
    st.sidebar.subheader("📅 Filtro de Período")
    
    # AQUI ESTÁ A MÁGICA: value=(min_date, max_date) seleciona tudo por padrão
    date_range = st.sidebar.date_input(
        "Selecione o intervalo:",
        value=(min_date, max_date), 
        min_value=min_date,
        max_value=max_date
    )

    # 3. Aplicação do Filtro
    df = raw_df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (raw_df['createdAt'].dt.date >= start_date) & (raw_df['createdAt'].dt.date <= end_date)
        df = raw_df.loc[mask].copy()

    # --- KPIs ---
    col1, col2, col3 = st.columns(3)
    
    total_pacientes = len(df)
    ativos = df[df["engagement_score"] > 0].copy()
    
    pct_ativos = (len(ativos) / total_pacientes * 100) if total_pacientes > 0 else 0

    col1.metric("👥 Pacientes Filtrados", total_pacientes)
    col2.metric("✅ Pacientes Ativos", len(ativos))
    col3.metric("📈 Engajamento", f"{pct_ativos:.1f}%")

    st.markdown("---")

    if total_pacientes == 0:
        st.warning(f"O filtro de data eliminou todos os registros.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Perfil", "Temporal", "Correlações"])

        # Aba 1: Visão Geral
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Distribuição")
                fig1 = plt.figure(figsize=(8, 4))
                sns.histplot(df["engagement_score"], bins=20, color='#9b59b6', kde=True)
                plt.xlabel("Interações")
                plt.ylabel("Qtd")
                sns.despine()
                st.pyplot(fig1, transparent=True)
            with c2:
                st.markdown("##### Funcionalidades")
                tipos = df[["n_symptoms", "n_acqs", "n_prescriptions", "n_activity_logs"]].sum()
                tipos_df = pd.DataFrame({"Func": ["Sintomas", "ACQ", "Meds", "Ativ."], "Total": tipos.values}).sort_values("Total", ascending=False)
                fig2 = plt.figure(figsize=(8, 4))
                sns.barplot(data=tipos_df, x="Func", y="Total", hue="Func", palette="BuPu", legend=False)
                sns.despine()
                st.pyplot(fig2, transparent=True)

        # Aba 2: Perfil
        with tab2:
            st.markdown("##### Análise de Gênero")
            df['sex_label'] = df['sex'].replace({'M': 'Masculino', 'F': 'Feminino'})
            ativos_sexo = df[df['engagement_score'] > 0].copy()
            ativos_sexo['sex_label'] = ativos_sexo['sex'].replace({'M': 'Masculino', 'F': 'Feminino'})

            if not ativos_sexo.empty:
                c1, c2, c3 = st.columns([1, 1, 1.5])
                total_sexo = ativos_sexo.groupby("sex_label")["engagement_score"].sum().reset_index()
                
                with c1:
                    st.markdown("**Média**")
                    fig3 = plt.figure(figsize=(4, 4))
                    sns.barplot(data=ativos_sexo, x="sex_label", y="engagement_score", hue="sex_label", palette="coolwarm", legend=False)
                    sns.despine()
                    st.pyplot(fig3, transparent=True)
                
                with c2:
                    st.markdown("**Proporção**")
                    fig_p = plt.figure(figsize=(4, 4))
                    colors = sns.color_palette("coolwarm", n_colors=2)
                    plt.pie(total_sexo["engagement_score"], labels=total_sexo["sex_label"], autopct='%1.0f%%', colors=colors, wedgeprops={'edgecolor': 'white'})
                    fig_p.gca().add_artist(plt.Circle((0,0),0.6,fc='#0E1117'))
                    st.pyplot(fig_p, transparent=True)
                
                with c3:
                    st.markdown("**Idade**")
                    fig4 = plt.figure(figsize=(6, 4))
                    sns.violinplot(data=ativos_sexo.dropna(subset=["age"]), x="sex_label", y="age", hue="sex_label", palette="coolwarm", legend=False)
                    sns.despine()
                    st.pyplot(fig4, transparent=True)

            st.divider()
            st.markdown("##### IMC")
            def categorizar_imc(bmi):
                if bmi < 18.5: return "Abaixo"
                elif bmi < 25: return "Normal"
                elif bmi < 30: return "Sobrepeso"
                else: return "Obesidade"
            
            ativos_imc = df[(df["engagement_score"] > 0) & (df["bmi"].notna())].copy()
            ativos_imc["bmi_category"] = ativos_imc["bmi"].apply(categorizar_imc)
            ordem = ["Abaixo", "Normal", "Sobrepeso", "Obesidade"]
            ativos_imc["bmi_category"] = pd.Categorical(ativos_imc["bmi_category"], categories=ordem, ordered=True)
            eng_imc = ativos_imc.groupby("bmi_category")["engagement_score"].mean().reset_index()
            
            fig5 = plt.figure(figsize=(10, 3))
            sns.barplot(data=eng_imc, x="bmi_category", y="engagement_score", hue="bmi_category", palette="Purples_d", legend=False)
            sns.despine()
            st.pyplot(fig5, transparent=True)

        # Aba 3: Temporal
        with tab3:
            Funcs = {"Sintomas": "symptomDiaries", "ACQ": "acqs", "Meds": "prescriptions", "Ativ.": "activityLogs"}
            lista_log = []
            for f, col in Funcs.items():
                for _, row in df.iterrows():
                    if isinstance(row[col], list):
                        for log in row[col]:
                            d = pd.to_datetime(log.get('createdAt'), errors='coerce')
                            if pd.notnull(d):
                                if isinstance(date_range, tuple) and len(date_range) == 2:
                                    if start_date <= d.date() <= end_date:
                                        lista_log.append({'date': d, 'Func': f})
                                else:
                                    lista_log.append({'date': d, 'Func': f})
            
            df_l = pd.DataFrame(lista_log)
            if not df_l.empty:
                df_l['Mês'] = df_l['date'].dt.to_period('M').astype(str)
                c_E, c_F = st.columns(2)
                with c_E:
                    st.markdown("##### Mensal")
                    fig6 = plt.figure(figsize=(8, 4))
                    sns.lineplot(data=df_l.groupby(['Mês', 'Func']).size().reset_index(name='T'), x='Mês', y='T', hue='Func', marker='o', palette="magma")
                    sns.despine()
                    st.pyplot(fig6, transparent=True)
                with c_F:
                    st.markdown("##### Heatmap")
                    df_l['dia'] = df_l['date'].dt.day_name().map({'Monday':'Seg','Tuesday':'Ter','Wednesday':'Qua','Thursday':'Qui','Friday':'Sex','Saturday':'Sáb','Sunday':'Dom'})
                    df_l['hora'] = df_l['date'].dt.hour
                    hm = df_l.groupby(['dia', 'hora']).size().unstack(fill_value=0)
                    fig_h = plt.figure(figsize=(8, 4))
                    sns.heatmap(hm, cmap="magma", cbar=False)
                    st.pyplot(fig_h, transparent=True)
            else:
                st.info("Sem dados temporais.")

        # Aba 4: Correlação
        with tab4:
            st.markdown("##### Correlação Idade vs Uso")
            df_c = df[(df["engagement_score"] > 0) & (df["age"].notna())].copy()
            if not df_c.empty:
                cols = {"n_symptoms": "Sint", "n_acqs": "ACQ", "n_prescriptions": "Meds", "n_activity_logs": "Ativ"}
                corrs = [{"Func": n, "r": df_c["age"].corr(df_c[c])} for c, n in cols.items()]
                fig8 = plt.figure(figsize=(8, 4))
                sns.barplot(data=pd.DataFrame(corrs), x="Func", y="r", hue="Func", palette="twilight", legend=False)
                plt.axhline(0, color='white', linewidth=0.5)
                sns.despine()
                st.pyplot(fig8, transparent=True)
else:
    st.info("Por favor, carregue o arquivo JSON na barra lateral.")

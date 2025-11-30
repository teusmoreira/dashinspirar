import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- DEFINIÇÃO DA NOVA IDENTIDADE VISUAL (ROXO & BRANCO) ---
# Cor principal (um roxo forte para contraste no branco)
PRIMARY_PURPLE = "#6A0DAD" 
# Cor secundária (um roxo mais claro para detalhes ou segunda categoria)
SECONDARY_PURPLE = "#9B59B6"

# Paleta monocromática (Correção aplicada aqui: usando light_palette)
PURPLE_PALETTE = sns.light_palette(PRIMARY_PURPLE, n_colors=5, reverse=True, input="hex")

# Configuração do Matplotlib para Fundo Branco e Texto Roxo
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    # Mudando todos os textos e eixos para roxo
    "text.color": PRIMARY_PURPLE,
    "axes.labelcolor": PRIMARY_PURPLE,
    "xtick.color": PRIMARY_PURPLE,
    "ytick.color": PRIMARY_PURPLE,
    "axes.edgecolor": PRIMARY_PURPLE,
    "font.size": 10,
    # Limpando as bordas dos gráficos para um visual mais "clean"
    "axes.spines.top": False,
    "axes.spines.right": False,
})
# -------------------------------------------------------

# --- Configuração da Página ---
st.set_page_config(page_title="Dashboard Inspirar", layout="wide", initial_sidebar_state="expanded")

# --- CSS CUSTOMIZADO PARA FORÇAR O TEMA BRANCO/ROXO NO STREAMLIT ---
st.markdown(f"""
    <style>
        /* Força fundo branco na aplicação principal e sidebar */
        .stApp, section[data-testid="stSidebar"] {{
            background-color: #FFFFFF;
        }}
        /* Muda a cor de todos os títulos e textos principais para roxo */
        h1, h2, h3, h4, p, span, label, .stDataFrame {{
            color: {PRIMARY_PURPLE} !important;
        }}
        /* Estiliza os cartões de métricas (KPIs) */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
             color: {PRIMARY_PURPLE} !important;
        }}
        [data-testid="stMetric"] {{
            background-color: #F8F0FF; /* Fundo roxo bem clarinho */
            border: 1px solid {SECONDARY_PURPLE};
            border-radius: 8px;
            padding: 10px;
        }}
        /* Cor dos divisores (hr) */
        hr {{
            border-color: {SECONDARY_PURPLE};
        }}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard de Engajamento - App Inspirar")
st.markdown("---") 

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📂 Fonte de Dados")
    uploaded_file = st.file_uploader("Carregar JSON", type=["json"])

# Caminho do arquivo local
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
        
        # Tratamentos de dados
        pacientes["createdAt"] = pd.to_datetime(pacientes["createdAt"], errors="coerce")
        pacientes["height"] = pd.to_numeric(pacientes["height"], errors='coerce')
        pacientes["height"] = np.where(pacientes["height"] > 3, pacientes["height"] / 100, pacientes["height"])
        
        # Cálculos de Scores
        pacientes["n_symptoms"] = pacientes["symptomDiaries"].apply(len)
        pacientes["n_acqs"] = pacientes["acqs"].apply(len)
        pacientes["n_prescriptions"] = pacientes["prescriptions"].apply(len)
        pacientes["n_activity_logs"] = pacientes["activityLogs"].apply(len)
        pacientes["engagement_score"] = (pacientes["n_symptoms"] + pacientes["n_acqs"] + 
                                         pacientes["n_prescriptions"] + pacientes["n_activity_logs"])
        
        # IMC
        pacientes["bmi"] = pacientes["weight"] / (pacientes["height"] ** 2)
        return pacientes
    except Exception as e:
        return None

# --- CARREGAMENTO ---
df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
elif df is None:
    try:
        df = load_data(LOCAL_PATH)
    except:
        pass

# --- VISUALIZAÇÃO ---
if df is not None:
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    
    total_pacientes = len(df)
    ativos = df[df["engagement_score"] > 0].copy()
    
    pct_ativos = (len(ativos) / total_pacientes * 100) if total_pacientes > 0 else 0

    col1.metric("👥 Total de Pacientes", total_pacientes)
    col2.metric("✅ Pacientes Ativos", len(ativos))
    col3.metric("📈 Taxa de Engajamento", f"{pct_ativos:.1f}%")

    st.markdown("---")

    # Abas
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Perfil", "Temporal", "Correlações"])

    # Aba 1: Visão Geral
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Distribuição de Interações")
            fig1 = plt.figure(figsize=(8, 4))
            # Mudança de cor para o roxo principal
            sns.histplot(df["engagement_score"], bins=20, color=PRIMARY_PURPLE, kde=True)
            plt.xlabel("Total Interações")
            plt.ylabel("Qtd Pacientes")
            sns.despine()
            st.pyplot(fig1, transparent=False) 
        with c2:
            st.markdown("##### Volume por Funcionalidade")
            tipos = df[["n_symptoms", "n_acqs", "n_prescriptions", "n_activity_logs"]].sum()
            tipos_df = pd.DataFrame({"Func": ["Sintomas", "ACQ", "Meds", "Ativ."], "Total": tipos.values}).sort_values("Total", ascending=False)
            fig2 = plt.figure(figsize=(8, 4))
            # Usando cor única roxa
            sns.barplot(data=tipos_df, x="Func", y="Total", color=PRIMARY_PURPLE)
            plt.ylabel("Registros")
            sns.despine()
            st.pyplot(fig2, transparent=False)

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
                sns.barplot(data=ativos_sexo, x="sex_label", y="engagement_score", color=PRIMARY_PURPLE)
                plt.xlabel("")
                plt.ylabel("")
                sns.despine()
                st.pyplot(fig3, transparent=False)
            
            with c2:
                st.markdown("**Proporção**")
                fig_p = plt.figure(figsize=(4, 4))
                colors = [PRIMARY_PURPLE, SECONDARY_PURPLE]
                plt.pie(total_sexo["engagement_score"], labels=total_sexo["sex_label"], autopct='%1.0f%%', colors=colors, wedgeprops={'edgecolor': 'white'})
                fig_p.gca().add_artist(plt.Circle((0,0),0.6,fc='white'))
                st.pyplot(fig_p, transparent=False)
            
            with c3:
                st.markdown("**Idade**")
                fig4 = plt.figure(figsize=(6, 4))
                sns.violinplot(data=ativos_sexo.dropna(subset=["age"]), x="sex_label", y="age", color=PRIMARY_PURPLE)
                plt.xlabel("")
                plt.ylabel("Idade")
                sns.despine()
                st.pyplot(fig4, transparent=False)

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
        sns.barplot(data=eng_imc, x="bmi_category", y="engagement_score", color=PRIMARY_PURPLE)
        plt.xlabel("")
        plt.ylabel("Engajamento Médio")
        sns.despine()
        st.pyplot(fig5, transparent=False)

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
                            lista_log.append({'date': d, 'Func': f})
        
        df_l = pd.DataFrame(lista_log)
        if not df_l.empty:
            df_l['Mês'] = df_l['date'].dt.to_period('M').astype(str)
            c_E, c_F = st.columns(2)
            with c_E:
                st.markdown("##### Mensal")
                fig6 = plt.figure(figsize=(8, 4))
                # Usando paleta monocromática corrigida
                sns.lineplot(data=df_l.groupby(['Mês', 'Func']).size().reset_index(name='T'), x='Mês', y='T', hue='Func', marker='o', palette=PURPLE_PALETTE)
                plt.grid(axis='y', alpha=0.3, linestyle='--', color=SECONDARY_PURPLE)
                sns.despine()
                st.pyplot(fig6, transparent=False)
            with c_F:
                st.markdown("##### Heatmap")
                df_l['dia'] = df_l['date'].dt.day_name().map({'Monday':'Seg','Tuesday':'Ter','Wednesday':'Qua','Thursday':'Qui','Friday':'Sex','Saturday':'Sáb','Sunday':'Dom'})
                df_l['hora'] = df_l['date'].dt.hour
                hm = df_l.groupby(['dia', 'hora']).size().unstack(fill_value=0)
                fig_h = plt.figure(figsize=(8, 4))
                cmap_purple = sns.light_palette(PRIMARY_PURPLE, as_cmap=True)
                sns.heatmap(hm, cmap=cmap_purple, cbar_kws={'label': 'Interações'}, linewidths=0.5, linecolor='white')
                plt.xlabel("Hora")
                plt.ylabel("")
                st.pyplot(fig_h, transparent=False)
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
            sns.barplot(data=pd.DataFrame(corrs), x="Func", y="r", color=PRIMARY_PURPLE)
            plt.axhline(0, color=PRIMARY_PURPLE, linewidth=0.5)
            plt.ylabel("Correlação (r)")
            sns.despine()
            st.pyplot(fig8, transparent=False)
else:
    st.info("Por favor, carregue o arquivo JSON na barra lateral ou verifique se o arquivo local existe.")

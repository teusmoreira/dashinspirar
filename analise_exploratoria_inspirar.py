import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. DEFINIÇÃO DAS CORES (Identidade Visual) ---
PRIMARY_PURPLE = "#6A0DAD"   # Roxo Forte
SECONDARY_PURPLE = "#9B59B6" # Roxo Médio
LIGHT_BG = "#FFFFFF"         # Branco

# --- 2. CONFIGURAÇÃO DOS GRÁFICOS ---
plt.rcParams.update({
    "figure.facecolor":  LIGHT_BG,
    "axes.facecolor":    LIGHT_BG,
    "savefig.facecolor": LIGHT_BG,
    "text.color":        PRIMARY_PURPLE,
    "axes.labelcolor":   PRIMARY_PURPLE,
    "xtick.color":       PRIMARY_PURPLE,
    "ytick.color":       PRIMARY_PURPLE,
    "axes.edgecolor":    PRIMARY_PURPLE,
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PURPLE_PALETTE = sns.light_palette(PRIMARY_PURPLE, n_colors=5, reverse=True, input="hex")

# --- 3. CONFIGURAÇÃO DA PÁGINA ---
# "collapsed" ajuda a esconder a barra se ela tentar aparecer vazia
st.set_page_config(page_title="Dashboard Inspirar", layout="wide", initial_sidebar_state="collapsed")

# --- 4. CSS (Tema Claro/Roxo) ---
st.markdown(f"""
    <style>
        .stApp, header[data-testid="stHeader"] {{
            background-color: {LIGHT_BG} !important;
        }}
        /* Esconde o botão de fechar sidebar já que não a usamos */
        [data-testid="collapsedControl"] {{
            display: none;
        }}
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, div, span, button {{
            color: {PRIMARY_PURPLE} !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {PRIMARY_PURPLE} !important;
            background-color: white !important;
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom-color: {PRIMARY_PURPLE} !important;
            font-weight: bold !important;
        }}
        [data-testid="stMetric"] {{
            background-color: #F8F0FF !important;
            border: 1px solid {SECONDARY_PURPLE};
            border-radius: 8px;
            padding: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
            color: {PRIMARY_PURPLE} !important;
        }}
        [data-testid="stFileUploader"] {{
            background-color: #F8F0FF;
            border-radius: 10px;
            padding: 10px;
        }}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard de Engajamento - App Inspirar")

# --- CONFIGURAÇÃO DE DADOS (NO TOPO DA PÁGINA) ---
# Usamos um Expander para o upload não poluir o visual principal
with st.expander("📂 Carregar novo arquivo de dados (JSON)", expanded=False):
    uploaded_file = st.file_uploader("Arraste seu arquivo aqui", type=["json"])

st.markdown("---") 

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
        
        # --- TRATAMENTO DE DADOS ---
        # 1. Datas
        pacientes["createdAt"] = pd.to_datetime(pacientes["createdAt"], errors="coerce")
        
        # 2. Altura (Limpeza robusta)
        pacientes["height"] = pacientes["height"].astype(str).str.replace(',', '.')
        pacientes["height"] = pd.to_numeric(pacientes["height"], errors='coerce')
        pacientes["height"] = np.where(pacientes["height"] > 3, pacientes["height"] / 100, pacientes["height"])
        
        # 3. Scores
        pacientes["n_symptoms"] = pacientes["symptomDiaries"].apply(len)
        pacientes["n_acqs"] = pacientes["acqs"].apply(len)
        pacientes["n_prescriptions"] = pacientes["prescriptions"].apply(len)
        pacientes["n_activity_logs"] = pacientes["activityLogs"].apply(len)
        pacientes["engagement_score"] = (pacientes["n_symptoms"] + pacientes["n_acqs"] + 
                                         pacientes["n_prescriptions"] + pacientes["n_activity_logs"])
        
        # 4. IMC
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
    
    # --- KPIs ---
    col1, col2, col3 = st.columns(3)
    total_pacientes = len(df)
    ativos = df[df["engagement_score"] > 0].copy()
    pct_ativos = (len(ativos) / total_pacientes * 100) if total_pacientes > 0 else 0

    col1.metric("👥 Total de Pacientes", total_pacientes)
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
                ax1 = sns.histplot(df["engagement_score"], bins=20, color=PRIMARY_PURPLE, kde=True)
                for p in ax1.patches:
                    if p.get_height() > 0:
                        ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='bottom', fontsize=9, color=PRIMARY_PURPLE, xytext=(0, 2),
                                     textcoords='offset points')
                plt.xlabel("Total Interações")
                plt.ylabel("Qtd")
                sns.despine()
                st.pyplot(fig1, transparent=False)
            with c2:
                st.markdown("##### Funcionalidades")
                tipos = df[["n_symptoms", "n_acqs", "n_prescriptions", "n_activity_logs"]].sum()
                tipos_df = pd.DataFrame({"Func": ["Sintomas", "ACQ", "Meds", "Ativ."], "Total": tipos.values}).sort_values("Total", ascending=False)
                fig2 = plt.figure(figsize=(8, 4))
                ax2 = sns.barplot(data=tipos_df, x="Func", y="Total", color=PRIMARY_PURPLE)
                ax2.bar_label(ax2.containers[0], fontsize=10, color=PRIMARY_PURPLE, padding=3)
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
                    ax3 = sns.barplot(data=ativos_sexo, x="sex_label", y="engagement_score", color=PRIMARY_PURPLE, errorbar=None)
                    ax3.bar_label(ax3.containers[0], fmt='%.1f', fontsize=10, color=PRIMARY_PURPLE, padding=3)
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
            ax5 = sns.barplot(data=eng_imc, x="bmi_category", y="engagement_score", color=PRIMARY_PURPLE)
            ax5.bar_label(ax5.containers[0], fmt='%.1f', fontsize=10, color=PRIMARY_PURPLE, padding=3)
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
                                # Fix Timezone
                                if d.tz is None: d = d.tz_localize('UTC')
                                d = d.tz_convert('America/Sao_Paulo')
                                # Sem filtro de data, adiciona tudo
                                lista_log.append({'date': d, 'Func': f})
            
            df_l = pd.DataFrame(lista_log)
            if not df_l.empty:
                df_l['Mês'] = df_l['date'].dt.to_period('M').astype(str)
                c_E, c_F = st.columns(2)
                with c_E:
                    st.markdown("##### Mensal")
                    fig6 = plt.figure(figsize=(8, 4))
                    ax6 = sns.lineplot(data=df_l.groupby(['Mês', 'Func']).size().reset_index(name='T'), 
                                       x='Mês', y='T', hue='Func', marker='o', palette=PURPLE_PALETTE)
                    for line in ax6.lines:
                        for x_val, y_val in zip(line.get_xdata(), line.get_ydata()):
                            ax6.text(x_val, y_val, f'{int(y_val)}', color=PRIMARY_PURPLE, fontsize=8)
                    plt.grid(axis='y', alpha=0.3, linestyle='--', color=SECONDARY_PURPLE)
                    sns.despine()
                    st.pyplot(fig6, transparent=False)
                with c_F:
                    st.markdown("##### Heatmap (Sem 00:00)")
                    # Filtra 00:00
                    df_hm = df_l[df_l['date'].dt.hour != 0].copy()
                    if not df_hm.empty:
                        mapa_dias = {'Monday':'Seg','Tuesday':'Ter','Wednesday':'Qua','Thursday':'Qui','Friday':'Sex','Saturday':'Sáb','Sunday':'Dom'}
                        df_hm['dia'] = df_hm['date'].dt.day_name().map(mapa_dias)
                        ordem_dias = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
                        df_hm['dia'] = pd.Categorical(df_hm['dia'], categories=ordem_dias, ordered=True)
                        df_hm['hora'] = df_hm['date'].dt.hour
                        
                        hm = df_hm.groupby(['dia', 'hora']).size().unstack(fill_value=0)
                        for h in range(1, 24):
                            if h not in hm.columns: hm[h] = 0
                        hm = hm.sort_index(axis=1)

                        fig_h = plt.figure(figsize=(8, 4))
                        cmap_purple = sns.light_palette(PRIMARY_PURPLE, as_cmap=True)
                        sns.heatmap(hm, cmap=cmap_purple, cbar_kws={'label': 'Interações'}, linewidths=0.5, linecolor='white')
                        plt.xlabel("Hora")
                        plt.ylabel("")
                        st.pyplot(fig_h, transparent=False)
                    else:
                        st.info("Apenas dados sem horário (00:00) encontrados.")
            else:
                st.info("Sem dados temporais.")

        # Aba 4: Correlação
        with tab4:
            st.markdown("##### Dispersão: Idade vs Engajamento")
            df_c = df[(df["engagement_score"] > 0) & (df["age"].notna())].copy()
            if not df_c.empty:
                c_sc, c_info = st.columns([3, 1])
                with c_sc:
                    fig8 = plt.figure(figsize=(10, 6))
                    sns.regplot(data=df_c, x="age", y="engagement_score", color=PRIMARY_PURPLE,
                                scatter_kws={'alpha': 0.6, 's': 60}, line_kws={'color': SECONDARY_PURPLE})
                    plt.xlabel("Idade")
                    plt.ylabel("Total Interações")
                    sns.despine()
                    plt.grid(alpha=0.2, linestyle='--')
                    st.pyplot(fig8, transparent=False)
                with c_info:
                    corr_val = df_c["age"].corr(df_c["engagement_score"])
                    st.metric("Correlação (r)", f"{corr_val:.3f}")
                    st.caption("Pontos = Pacientes\nLinha = Tendência")
            else:
                st.warning("Dados insuficientes.")
else:
    st.info("Por favor, carregue o arquivo JSON no topo da página ou verifique se o arquivo local existe.")

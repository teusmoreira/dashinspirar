import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURAÇÃO VISUAL (TEMA ESCURO E TRANSPARENTE) ---
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
# -------------------------------------------------------

# --- Configuração da Página ---
st.set_page_config(page_title="Dashboard Inspirar", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Dashboard de Engajamento - App Inspirar")
st.markdown("---") 

# --- Barra Lateral para Upload ---
with st.sidebar:
    st.header("📂 Fonte de Dados")
    uploaded_file = st.file_uploader("Carregar JSON", type=["json"])
    st.info("Faça o upload do arquivo 'pacientes...json' para visualizar os dados atualizados.")

# Caminho fixo local
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
        
        # Tratamento de dados
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
        
        # IMC
        pacientes["bmi"] = pacientes["weight"] / (pacientes["height"] ** 2)
        return pacientes
    except Exception as e:
        return None



raw_df = None
if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
else:
    try:
        raw_df = load_data(LOCAL_PATH)
        # Removi os st.sidebar.success/warning daqui como pediu
    except:
        pass

# --- LÓGICA DE FILTRO DE DATA ---
df = None
if raw_df is not None:
    # 1. Pega data min/max dos dados carregados
    min_date = raw_df["createdAt"].min().date()
    max_date = raw_df["createdAt"].max().date()

    # 2. Cria o filtro na sidebar
    with st.sidebar:
        st.divider()
        st.subheader("📅 Filtro de Período")
        date_range = st.date_input(
            "Selecione o intervalo:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    # 3. Aplica o filtro
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (raw_df['createdAt'].dt.date >= start_date) & (raw_df['createdAt'].dt.date <= end_date)
        df = raw_df.loc[mask].copy()
    else:
        df = raw_df.copy()

# --- CONSTRUÇÃO DO DASHBOARD ---
if df is not None:
    # KPI Section
    col1, col2, col3 = st.columns(3)
    
    total_pacientes = len(df)
    ativos = df[df["engagement_score"] > 0].copy()
    pct_ativos = (len(ativos) / total_pacientes) * 100

    col1.metric("👥 Total de Pacientes", total_pacientes)
    col2.metric("✅ Pacientes Ativos", len(ativos), help="Pacientes com pelo menos 1 interação")
    col3.metric("📈 Taxa de Engajamento", f"{pct_ativos:.1f}%")

    st.markdown("---")

    # Abas organizadas
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Visão Geral", 
        "users Demografia & IMC", 
        "📅 Análise Temporal", 
        "🔗 Correlações"
    ])

    # --- ABA 1 ---
    with tab1:
        col_A, col_B = st.columns(2)
        
        with col_A:
            st.markdown("##### Distribuição de Interações")
            fig1 = plt.figure(figsize=(8, 4))
            sns.histplot(df["engagement_score"], bins=20, color='#9b59b6', kde=True)
            plt.ylabel("Qtd. Pacientes")
            plt.xlabel("Total Interações")
            sns.despine()
            st.pyplot(fig1, transparent=True)

        with col_B:
            st.markdown("##### Total por Funcionalidade")
            tipos = df[["n_symptoms", "n_acqs", "n_prescriptions", "n_activity_logs"]].sum()
            tipos_df = pd.DataFrame({
                "Funcionalidade": ["Sintomas", "ACQ", "Medicamentos", "Ativ. Físicas"],
                "Interações": tipos.values
            }).sort_values(by="Interações", ascending=False)

            fig2 = plt.figure(figsize=(8, 4))
            sns.barplot(data=tipos_df, x="Funcionalidade", y="Interações", hue="Funcionalidade", palette="BuPu", legend=False)
            plt.ylabel("Registros")
            sns.despine()
            st.pyplot(fig2, transparent=True)

    # --- ABA 2 ---
    with tab2:
        st.markdown("##### Análise de Gênero: Média vs. Volume Total")
        
        # Dados e Tradução
        df['sex_label'] = df['sex'].replace({'M': 'Masculino', 'F': 'Feminino'})
        ativos_sexo = df[df['engagement_score'] > 0].copy()
        ativos_sexo['sex_label'] = ativos_sexo['sex'].replace({'M': 'Masculino', 'F': 'Feminino'})
        
        # Agrupamento Soma
        total_por_sexo = ativos_sexo.groupby("sex_label")["engagement_score"].sum().reset_index()
        
        col_C, col_D, col_E = st.columns([1, 1, 1.5])
        
        with col_C:
            st.markdown("**Média de Interações**")
            fig3 = plt.figure(figsize=(5, 5))
            sns.barplot(data=ativos_sexo, x="sex_label", y="engagement_score", hue="sex_label", palette="coolwarm", legend=False)
            plt.ylabel("")
            plt.xlabel("")
            sns.despine()
            st.pyplot(fig3, transparent=True)

        with col_D:
            st.markdown("**Volume Total (Proporção)**")
            fig_pizza = plt.figure(figsize=(5, 5))
            
            # Define cores baseado na ordem dos dados para garantir coerência (Azul p/ Masc, Vermelho/Rosa p/ Fem)
            colors = sns.color_palette("coolwarm", n_colors=2)
            
            plt.pie(
                total_por_sexo["engagement_score"], 
                labels=total_por_sexo["sex_label"], 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            # Furo no meio (Donut)
            centre_circle = plt.Circle((0,0),0.60,fc='#0E1117') # Cor aproximada do fundo do Streamlit Dark
            fig_pizza.gca().add_artist(centre_circle)
            
            st.pyplot(fig_pizza, transparent=True)

        with col_E:
            st.markdown("**Distribuição de Idade**")
            ativos_validos = ativos_sexo.dropna(subset=["age"]).copy()
            fig4 = plt.figure(figsize=(6, 5))
            sns.violinplot(
                data=ativos_validos, 
                x="sex_label", 
                y="age", 
                hue="sex_label", 
                palette="coolwarm", 
                inner="stick", 
                legend=False
            )
            plt.xlabel("")
            plt.ylabel("Idade")
            sns.despine()
            st.pyplot(fig4, transparent=True)

        st.divider()
        st.markdown("##### Engajamento por Categoria de IMC")
        def categorizar_imc(bmi):
            if bmi < 18.5: return "Abaixo (<18.5)"
            elif bmi < 25: return "Normal (18.5-24.9)"
            elif bmi < 30: return "Sobrepeso (25-29.9)"
            else: return "Obesidade (>=30)"

        ativos_imc = df[(df["engagement_score"] > 0) & (df["bmi"].notna())].copy()
        ativos_imc["bmi_category"] = ativos_imc["bmi"].apply(categorizar_imc)
        
        ordem = ["Abaixo (<18.5)", "Normal (18.5-24.9)", "Sobrepeso (25-29.9)", "Obesidade (>=30)"]
        ativos_imc["bmi_category"] = pd.Categorical(ativos_imc["bmi_category"], categories=ordem, ordered=True)
        
        engajamento_imc = ativos_imc.groupby("bmi_category")["engagement_score"].mean().reset_index()

        fig5 = plt.figure(figsize=(12, 4))
        sns.barplot(data=engajamento_imc, x="bmi_category", y="engagement_score", hue="bmi_category", palette="Purples_d", legend=False)
        plt.xlabel("Categoria IMC")
        plt.ylabel("Engajamento Médio")
        sns.despine()
        st.pyplot(fig5, transparent=True)

    # --- ABA 3 ---
    with tab3:
        # Prepara dados básicos
        Funcionalidades = {"Sintomas": "symptomDiaries", "ACQ": "acqs", "Medicamentos": "prescriptions", "Atividades": "activityLogs"}
        lista_log = []
        for feat_name, col_name in Funcionalidades.items():
            for _, row in df.iterrows():
                if isinstance(row[col_name], list):
                    for log in row[col_name]:
                        lista_log.append({'date': pd.to_datetime(log.get('createdAt'), errors='coerce'), 'Funcionalidade': feat_name})
        
        df_logs = pd.DataFrame(lista_log).dropna()

        if not df_logs.empty:
            col_E, col_F = st.columns(2)

            with col_E:
                st.markdown("##### Evolução Mensal")
                df_logs['Mês'] = df_logs['date'].dt.to_period('M').astype(str)
                mensal = df_logs.groupby(['Mês', 'Funcionalidade']).size().reset_index(name='Total')
                fig6 = plt.figure(figsize=(8, 4))
                sns.lineplot(data=mensal, x='Mês', y='Total', hue='Funcionalidade', marker='o', palette="magma", linewidth=2.5)
                plt.grid(axis='y', alpha=0.3, linestyle='--')
                sns.despine()
                st.pyplot(fig6, transparent=True)

            with col_F:
                st.markdown("##### Padrão Semanal (Sintomas)")
                logs_sint = df_logs[df_logs['Funcionalidade'] == 'Sintomas'].copy()
                logs_sint['dia_num'] = logs_sint['date'].dt.dayofweek
                dias = {0:'Seg', 1:'Ter', 2:'Qua', 3:'Qui', 4:'Sex', 5:'Sáb', 6:'Dom'}
                logs_sint['Dia'] = logs_sint['dia_num'].map(dias)
                semanal = logs_sint.groupby(['Dia', 'dia_num']).size().reset_index(name='Total').sort_values('dia_num')
                fig7 = plt.figure(figsize=(8, 4))
                sns.barplot(data=semanal, x='Dia', y='Total', palette="BuPu", hue="Dia", legend=False)
                sns.despine()
                st.pyplot(fig7, transparent=True)
            
            # --- MAPA DE CALOR ---
            st.divider()
            st.subheader("🔥 Mapa de Calor: Concentração de Uso")

            all_logs_heat = []
            for col in ["symptomDiaries", "acqs", "prescriptions", "activityLogs"]:
                for logs in df[col]:
                    if isinstance(logs, list):
                        for entry in logs:
                            if 'createdAt' in entry:
                                all_logs_heat.append(entry['createdAt'])

            df_time = pd.DataFrame(all_logs_heat, columns=['datetime'])
            df_time['datetime'] = pd.to_datetime(df_time['datetime'])
            df_time['dia_semana'] = df_time['datetime'].dt.day_name()
            df_time['hora'] = df_time['datetime'].dt.hour
            dias_traducao = {'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta', 'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
            df_time['dia_semana'] = df_time['dia_semana'].map(dias_traducao)
            ordem_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
            df_time['dia_semana'] = pd.Categorical(df_time['dia_semana'], categories=ordem_dias, ordered=True)
            heatmap_data = df_time.groupby(['dia_semana', 'hora']).size().unstack(fill_value=0)

            fig_heat = plt.figure(figsize=(12, 5))
            sns.heatmap(heatmap_data, cmap="magma", linewidths=.5, linecolor='#1E1E1E', cbar_kws={'label': 'Total Interações'})
            plt.title("Intensidade de uso por Dia e Hora")
            plt.xlabel("Hora do Dia (0-23h)")
            plt.ylabel("")
            st.pyplot(fig_heat, transparent=True)

        else:
            st.info("Sem dados temporais suficientes.")

    # --- ABA 4 ---
    with tab4:
        st.markdown("##### Correlação: Idade vs Frequência de Uso")
        df_corr = df[(df["engagement_score"] > 0) & (df["age"].notna())].copy()
        
        cols_map = {"n_symptoms": "Sintomas", "n_acqs": "ACQ", "n_prescriptions": "Meds", "n_activity_logs": "Ativ. Física"}
        corrs = [{"Funcionalidade": nome, "r": df_corr["age"].corr(df_corr[col])} for col, nome in cols_map.items()]
        df_corrs = pd.DataFrame(corrs).sort_values("r", ascending=False)

        col_G, col_H = st.columns([2, 1])
        
        with col_G:
            fig8 = plt.figure(figsize=(8, 5))
            sns.barplot(data=df_corrs, x="Funcionalidade", y="r", hue="Funcionalidade", palette="twilight", legend=False)
            plt.axhline(0, color='white', linewidth=0.5)
            plt.ylabel("Correlação de Pearson (r)")
            sns.despine()
            st.pyplot(fig8, transparent=True)

        with col_H:
            st.dataframe(df_corrs.style.format({"r": "{:.4f}"}), hide_index=True, use_container_width=True)
            st.caption("Nota: Correlações positivas indicam maior uso com o aumento da idade.")

else:
    st.info("Por favor, carregue o arquivo JSON na barra lateral para começar.")

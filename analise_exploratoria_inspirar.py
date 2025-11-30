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

# --- Configuração da Página ---
st.set_page_config(page_title="Dashboard Inspirar", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Dashboard de Engajamento - App Inspirar")
st.markdown("---") 

# --- BARRA LATERAL (LIMPA & COM FILTRO) ---
with st.sidebar:
    st.header("Configurações")
    uploaded_file = st.file_uploader("Carregar JSON", type=["json"])
    # Removi os avisos de texto (st.info/st.success) daqui

# Caminho fixo local (Tenta ler dados_demo.json se existir, senão o original)
LOCAL_PATH = "dados_demo.json" 

@st.cache_data
def load_data(file_input):
    try:
        if isinstance(file_input, str):
            with open(file_input, "r", encoding="utf-8") as dataset:
                data = json.load(dataset)
        else:
            data = json.load(file_input)

        pacientes = pd.json_normalize(data["data"]["result"])
        
        # Tratamento
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

# Lógica de Carregamento (Silenciosa - sem avisos na sidebar)
raw_df = None
if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
else:
    try:
        raw_df = load_data(LOCAL_PATH)
    except:
        pass # Falha silenciosa se não achar arquivo local

# --- LÓGICA DE FILTRO E EXIBIÇÃO ---
if raw_df is not None:
    
    # 1. Detectar datas do arquivo para configurar o filtro
    min_date = raw_df["createdAt"].min().date()
    max_date = raw_df["createdAt"].max().date()

    # 2. Criar o Filtro na Barra Lateral
    with st.sidebar:
        st.divider()
        st.subheader("📅 Filtro de Período")
        
        date_range = st.date_input(
            "Selecione o intervalo:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    # 3. Aplicar o Filtro ao DataFrame Principal
    # Se o usuário selecionar um intervalo válido (2 datas)
    if len(date_range) == 2:
        start_date, end_date = date_range
        # Filtra pacientes criados neste período
        mask = (raw_df['createdAt'].dt.date >= start_date) & (raw_df['createdAt'].dt.date <= end_date)
        df = raw_df.loc[mask].copy()
    else:
        df = raw_df.copy()

    # --- KPI Section ---
    col1, col2, col3 = st.columns(3)
    
    total_pacientes = len(df)
    ativos = df[df["engagement_score"] > 0].copy()
    
    if total_pacientes > 0:
        pct_ativos = (len(ativos) / total_pacientes) * 100
    else:
        pct_ativos = 0

    col1.metric("👥 Pacientes Filtrados", total_pacientes)
    col2.metric("✅ Pacientes Ativos", len(ativos))
    col3.metric("📈 Engajamento", f"{pct_ativos:.1f}%")

    st.markdown("---")

    if total_pacientes == 0:
        st.warning("⚠️ Nenhum dado encontrado para o período selecionado.")
    else:
        # Abas
        tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Perfil", "Temporal", "Correlações"])

        # --- ABA 1: Visão Geral ---
        with tab1:
            col_A, col_B = st.columns(2)
            with col_A:
                st.markdown("##### Distribuição de Interações")
                fig1 = plt.figure(figsize=(8, 4))
                sns.histplot(df["engagement_score"], bins=20, color='#9b59b6', kde=True)
                plt.xlabel("Total Interações")
                plt.ylabel("Qtd Pacientes")
                sns.despine()
                st.pyplot(fig1, transparent=True)
            with col_B:
                st.markdown("##### Volume por Funcionalidade")
                tipos = df[["n_symptoms", "n_acqs", "n_prescriptions", "n_activity_logs"]].sum()
                tipos_df = pd.DataFrame({
                    "Func": ["Sintomas", "ACQ", "Meds", "Ativ. Física"],
                    "Total": tipos.values
                }).sort_values("Total", ascending=False)
                fig2 = plt.figure(figsize=(8, 4))
                sns.barplot(data=tipos_df, x="Func", y="Total", hue="Func", palette="BuPu", legend=False)
                plt.ylabel("Registros")
                sns.despine()
                st.pyplot(fig2, transparent=True)

        # --- ABA 2: Perfil ---
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
                    plt.xlabel("")
                    plt.ylabel("")
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
                    ativos_validos = ativos_sexo.dropna(subset=["age"]).copy()
                    fig4 = plt.figure(figsize=(6, 4))
                    sns.violinplot(data=ativos_validos, x="sex_label", y="age", hue="sex_label", palette="coolwarm", legend=False, inner="stick")
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
            plt.xlabel("")
            plt.ylabel("Engajamento Médio")
            sns.despine()
            st.pyplot(fig5, transparent=True)

        # --- ABA 3: Temporal ---
        with tab3:
            # Extração de logs com filtro de data aplicado aos LOGS individuais também
            Funcionalidades = {"Sintomas": "symptomDiaries", "ACQ": "acqs", "Meds": "prescriptions", "Ativ.": "activityLogs"}
            lista_log = []
            
            for feat, col in Funcionalidades.items():
                for _, row in df.iterrows():
                    if isinstance(row[col], list):
                        for log in row[col]:
                            d_log = pd.to_datetime(log.get('createdAt'), errors='coerce')
                            if pd.notnull(d_log):
                                # Se houver filtro de data, verificamos se o log está dentro
                                if len(date_range) == 2:
                                    if start_date <= d_log.date() <= end_date:
                                        lista_log.append({'date': d_log, 'Func': feat})
                                else:
                                    lista_log.append({'date': d_log, 'Func': feat})

            df_logs = pd.DataFrame(lista_log)

            if not df_logs.empty:
                col_E, col_F = st.columns(2)
                with col_E:
                    st.markdown("##### Evolução Mensal")
                    df_logs['Mês'] = df_logs['date'].dt.to_period('M').astype(str)
                    mensal = df_logs.groupby(['Mês', 'Func']).size().reset_index(name='Total')
                    fig6 = plt.figure(figsize=(8, 4))
                    sns.lineplot(data=mensal, x='Mês', y='Total', hue='Func', marker='o', palette="magma", linewidth=2.5)
                    plt.grid(axis='y', alpha=0.3, linestyle='--')
                    sns.despine()
                    st.pyplot(fig6, transparent=True)
                
                with col_F:
                    st.markdown("##### Padrão Semanal (Sintomas)")
                    logs_sint = df_logs[df_logs['Func'] == 'Sintomas'].copy()
                    logs_sint['dia_num'] = logs_sint['date'].dt.dayofweek
                    dias = {0:'Seg', 1:'Ter', 2:'Qua', 3:'Qui', 4:'Sex', 5:'Sáb', 6:'Dom'}
                    logs_sint['Dia'] = logs_sint['dia_num'].map(dias)
                    semanal = logs_sint.groupby(['Dia', 'dia_num']).size().reset_index(name='Total').sort_values('dia_num')
                    fig7 = plt.figure(figsize=(8, 4))
                    sns.barplot(data=semanal, x='Dia', y='Total', palette="BuPu", hue="Dia", legend=False)
                    sns.despine()
                    st.pyplot(fig7, transparent=True)

                st.divider()
                st.subheader("🔥 Mapa de Calor")
                
                df_logs['dia_semana'] = df_logs['date'].dt.day_name()
                df_logs['hora'] = df_logs['date'].dt.hour
                dias_traducao = {'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta', 'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
                df_logs['dia_semana'] = df_logs['dia_semana'].map(dias_traducao)
                ordem_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
                df_logs['dia_semana'] = pd.Categorical(df_logs['dia_semana'], categories=ordem_dias, ordered=True)
                heatmap_data = df_logs.groupby(['dia_semana', 'hora']).size().unstack(fill_value=0)

                fig_heat = plt.figure(figsize=(12, 5))
                sns.heatmap(heatmap_data, cmap="magma", linewidths=.5, linecolor='#1E1E1E', cbar_kws={'label': 'Interações'})
                plt.xlabel("Hora do Dia")
                plt.ylabel("")
                st.pyplot(fig_heat, transparent=True)
            else:
                st.info("Sem dados temporais no período selecionado.")

        # --- ABA 4: Correlações ---
        with tab4:
            st.markdown("##### Correlação: Idade vs Frequência de Uso")
            df_corr = df[(df["engagement_score"] > 0) & (df["age"].notna())].copy()
            
            if not df_corr.empty:
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
            else:
                st.info("Dados insuficientes para correlação.")
else:
    # Se não houver dados carregados, mostra apenas um aviso limpo
    st.info("Por favor, carregue o arquivo JSON na barra lateral.")

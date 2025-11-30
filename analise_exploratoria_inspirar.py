import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuração da Página ---
st.set_page_config(page_title="Dashboard Inspirar", layout="wide")

st.title("📊 Dashboard de Engajamento - App Inspirar")
st.markdown("Análise de dados de pacientes entre Março e Julho.")

# --- Barra Lateral para Upload ---
st.sidebar.header("Carregar Dados")
uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo JSON", type=["json"])

# Caminho fixo local (fallback caso não queira usar o uploader)
LOCAL_PATH = "/pacientes_marco-julho_com_createdAt_com_sexo_sigla_filtrado.json"

@st.cache_data
def load_data(file_input):
    """Função para carregar e processar os dados iniciais"""
    try:
        # Verifica se é um arquivo enviado pelo Streamlit ou caminho local
        if isinstance(file_input, str):
            with open(file_input, "r", encoding="utf-8") as dataset:
                data = json.load(dataset)
        else:
            data = json.load(file_input)

        pacientes = pd.json_normalize(data["data"]["result"])
        
        # Conversões
        pacientes["createdAt"] = pd.to_datetime(pacientes["createdAt"], errors="coerce")
        pacientes["height"] = pd.to_numeric(pacientes["height"], errors='coerce')
        
        # Correção de Altura
        pacientes["height"] = np.where(
            pacientes["height"] > 3,
            pacientes["height"] / 100,
            pacientes["height"]
        )
        
        # Cálculo de Scores
        pacientes["n_symptoms"] = pacientes["symptomDiaries"].apply(len)
        pacientes["n_acqs"] = pacientes["acqs"].apply(len)
        pacientes["n_prescriptions"] = pacientes["prescriptions"].apply(len)
        pacientes["n_activity_logs"] = pacientes["activityLogs"].apply(len)

        pacientes["engagement_score"] = (
            pacientes["n_symptoms"]
            + pacientes["n_acqs"]
            + pacientes["n_prescriptions"]
            + pacientes["n_activity_logs"]
        )
        
        # Cálculo de IMC
        pacientes["bmi"] = pacientes["weight"] / (pacientes["height"] ** 2)
        
        return pacientes

    except Exception as e:
        return None

# Lógica de Carregamento
df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    # Tenta carregar do caminho local se o usuário não fez upload
    try:
        df = load_data(LOCAL_PATH)
        st.sidebar.success(f"Arquivo local carregado: {LOCAL_PATH}")
    except:
        st.warning("Aguardando arquivo JSON. Por favor, faça o upload na barra lateral.")

if df is not None:
    # --- Métricas Principais (KPIs) ---
    total_pacientes = len(df)
    ativos = df[df["engagement_score"] > 0].copy()
    total_ativos = len(ativos)
    pct_ativos = (total_ativos / total_pacientes) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Pacientes", total_pacientes)
    col2.metric("Pacientes Ativos (>1 interação)", total_ativos)
    col3.metric("Taxa de Atividade", f"{pct_ativos:.1f}%")

    st.divider()

    # --- Criação de Abas para Organização ---
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Demografia & IMC", "Análise Temporal", "Correlações"])

    # ==========================================================
    # ABA 1: VISÃO GERAL
    # ==========================================================
    with tab1:
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Distribuição do Engajamento")
            fig1 = plt.figure(figsize=(8, 4))
            sns.histplot(
                df["engagement_score"],
                bins=20,
                color=sns.color_palette("Purples", n_colors=6)[5]
            )
            plt.title("Histograma de Interações")
            plt.xlabel("Número total de interações")
            plt.ylabel("Número de pacientes")
            st.pyplot(fig1)

        with c2:
            st.subheader("Volume por Funcionalidade")
            tipos = df[["n_symptoms", "n_acqs", "n_prescriptions", "n_activity_logs"]].sum()
            tipos_df = pd.DataFrame({
                "Funcionalidade": [
                    "Diário de Sintomas", "Questionário ACQ", 
                    "Medicamentos", "Atividades Físicas"
                ],
                "Interações": tipos.values
            }).sort_values(by="Interações", ascending=False)

            fig2 = plt.figure(figsize=(8, 5))
            sns.barplot(
                data=tipos_df, x="Funcionalidade", y="Interações",
                hue="Funcionalidade", palette=("BuPu"), legend=False
            )
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Total de Registros")
            st.pyplot(fig2)

    # ==========================================================
    # ABA 2: DEMOGRAFIA & IMC
    # ==========================================================
    with tab2:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### Engajamento por Gênero")
            fig3 = plt.figure(figsize=(6, 4))
            sns.barplot(
                data=ativos, x="sex", y="engagement_score",
                hue="engagement_score", palette=("flare"), legend=False
            )
            plt.xlabel("Gênero")
            plt.ylabel("Interações")
            st.pyplot(fig3)

        with c2:
            st.markdown("### Idade vs Engajamento (Violin Plot)")
            ativos_validos = ativos.dropna(subset=["age", "sex"]).copy()
            fig4 = plt.figure(figsize=(8, 5))
            sns.violinplot(
                data=ativos_validos, x="sex", y="age", hue="sex",
                palette="magma", legend=False, inner=None, alpha=0.35
            )
            sns.stripplot(
                data=ativos_validos, x="sex", y="age", hue="sex",
                palette="magma", legend=False, dodge=False, jitter=0.15, size=5, alpha=0.85
            )
            sns.despine()
            st.pyplot(fig4)

        st.divider()
        st.markdown("### Análise por IMC (BMI)")
        
        def categorizar_imc(bmi):
            if bmi < 18.5: return "Abaixo do Peso (<18.5)"
            elif bmi < 25: return "Peso Normal (18.5 - 24.9)"
            elif bmi < 30: return "Sobrepeso (25.0 - 29.9)"
            else: return "Obesidade (>=30.0)"

        ativos_imc = df[(df["engagement_score"] > 0) & (df["bmi"].notna())].copy()
        ativos_imc["bmi_category"] = ativos_imc["bmi"].apply(categorizar_imc)
        
        engajamento_por_imc = ativos_imc.groupby("bmi_category")["engagement_score"].mean().reset_index(name="Engajamento_Médio")
        ordem_imc = ["Abaixo do Peso (<18.5)", "Peso Normal (18.5 - 24.9)", "Sobrepeso (25.0 - 29.9)", "Obesidade (>=30.0)"]
        engajamento_por_imc["bmi_category"] = pd.Categorical(engajamento_por_imc["bmi_category"], categories=ordem_imc, ordered=True)
        engajamento_por_imc = engajamento_por_imc.sort_values("bmi_category")

        fig5 = plt.figure(figsize=(10, 5))
        sns.barplot(
            data=engajamento_por_imc, x="bmi_category", y="Engajamento_Médio",
            hue="Engajamento_Médio", palette="Purples_d", legend=False
        )
        plt.xticks(rotation=15, ha='right')
        st.pyplot(fig5)

    # ==========================================================
    # ABA 3: ANÁLISE TEMPORAL
    # ==========================================================
    with tab3:
        # Funções de processamento de data
        Funcionalidades = {
            "Diário de Sintomas": "symptomDiaries",
            "Questionário ACQ": "acqs",
            "Medicamentos": "prescriptions",
            "Atividades Físicas": "activityLogs"
        }

        def disaggregate_logs(df_input, log_column, feature_name):
            lista_log = []
            for _, row in df_input.iterrows():
                if isinstance(row[log_column], list):
                    for log in row[log_column]:
                        lista_log.append({
                            'date': pd.to_datetime(log.get('createdAt'), errors='coerce'),
                            'Funcionalidade': feature_name
                        })
            return pd.DataFrame(lista_log).dropna(subset=['date'])

        # Processamento
        total_logs_dfs = []
        for feature_name, column_name in Funcionalidades.items():
            df_desagregado = disaggregate_logs(df, column_name, feature_name)
            total_logs_dfs.append(df_desagregado)
        
        if total_logs_dfs:
            logs_combinados = pd.concat(total_logs_dfs)
            logs_combinados['Mês'] = logs_combinados['date'].dt.tz_localize(None).dt.to_period('M')
            engajamento_mensal = logs_combinados.groupby(['Mês', 'Funcionalidade']).size().reset_index(name='Total_Registros')
            engajamento_mensal['Mês'] = engajamento_mensal['Mês'].astype(str)

            st.markdown("### Evolução Mensal")
            fig6 = plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=engajamento_mensal, x='Mês', y='Total_Registros',
                hue='Funcionalidade', marker='o', palette=sns.color_palette("magma", n_colors=4)
            )
            plt.grid(axis='y', alpha=0.5)
            st.pyplot(fig6)

            st.markdown("### Padrão Semanal (Diário de Sintomas)")
            logs_sintomas = logs_combinados[logs_combinados['Funcionalidade'] == 'Diário de Sintomas'].copy()
            logs_sintomas['dia_semana_num'] = logs_sintomas['date'].dt.tz_localize(None).dt.dayofweek
            dias_semana = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
            logs_sintomas['Dia_da_Semana'] = logs_sintomas['dia_semana_num'].map(dias_semana)

            engajamento_semanal = logs_sintomas.groupby(['Dia_da_Semana', 'dia_semana_num']).size().reset_index(name='Total_Registros')
            engajamento_semanal = engajamento_semanal.sort_values('dia_semana_num')

            fig7 = plt.figure(figsize=(10, 5))
            sns.barplot(
                data=engajamento_semanal, x='Dia_da_Semana', y='Total_Registros',
                palette='BuPu', hue='Dia_da_Semana', legend=False
            )
            st.pyplot(fig7)
        else:
            st.warning("Não há logs suficientes para análise temporal.")

    # ==========================================================
    # ABA 4: CORRELAÇÕES
    # ==========================================================
    with tab4:
        st.markdown("### Correlação de Pearson: Idade vs. Uso")
        
        df_ativos_validos = df[
            (df["engagement_score"] > 0) & (df["age"].notna())
        ].copy()

        colunas_uso = ["n_symptoms", "n_acqs", "n_prescriptions", "n_activity_logs"]
        nomes_colunas = {
            "n_symptoms": "Diário de Sintomas",
            "n_acqs": "Questionário ACQ",
            "n_prescriptions": "Medicamentos",
            "n_activity_logs": "Atividades Físicas",
        }

        lista_correlacao = []
        for coluna in colunas_uso:
            correlacao = df_ativos_validos["age"].corr(df_ativos_validos[coluna])
            lista_correlacao.append({
                "Funcionalidade": nomes_colunas[coluna],
                "Correlacao (r)": correlacao
            })

        correlacoes_df = pd.DataFrame(lista_correlacao).sort_values(by="Correlacao (r)", ascending=False)

        c1, c2 = st.columns([2, 1])
        
        with c1:
            fig8 = plt.figure(figsize=(10, 6))
            sns.barplot(
                data=correlacoes_df, x="Funcionalidade", y="Correlacao (r)",
                hue="Funcionalidade", palette="twilight", legend=False
            )
            plt.axhline(0, color='black', linewidth=0.8)
            plt.ylabel("Coeficiente de Correlação (r)")
            st.pyplot(fig8)

        with c2:
            st.markdown("**Tabela de Dados**")
            st.dataframe(correlacoes_df, hide_index=True)
            st.info("""
            **Interpretação:**
            * **r > 0**: Aumenta com a idade.
            * **r < 0**: Diminui com a idade.
            * **r ≈ 0**: Sem correlação clara.
            """)

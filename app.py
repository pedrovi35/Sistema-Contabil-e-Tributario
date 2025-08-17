# app.py - Sistema Fiscal Inteligente com Agentes Especializados (Versão Completa com Agentes 2.0)

# --- Importações Essenciais ---
import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import ollama
import os
from datetime import datetime
import re

# --- Importações para Funcionalidades Adicionais ---
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from fpdf import FPDF
import feedparser

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(layout="wide", page_title="Plataforma de Agentes Fiscais 2.0")

# ==============================================================================
# MÓDULO 1: INGESTÃO DE DADOS
# ==============================================================================
def parse_nfe_xml(xml_file):
    """Lê um arquivo XML de NF-e e extrai informações chave, incluindo dados para análise tributária."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        ns = {'nfe': 'http://www.portalfiscal.inf.br/nfe'}
        
        # Encontrar os nós principais
        ide = root.find('.//nfe:ide', ns)
        emit = root.find('.//nfe:emit/nfe:enderEmit', ns)
        dest = root.find('.//nfe:dest/nfe:enderDest', ns)
        total = root.find('.//nfe:ICMSTot', ns)
        emit_info = root.find('.//nfe:emit', ns)
        dest_info = root.find('.//nfe:dest', ns)

        # Helper para extrair texto de um nó
        def find_text(node, path, default=None):
            if node is None: return default
            found_node = node.find(path, ns)
            return found_node.text if found_node is not None else default

        # Helper para extrair float de um nó
        def find_float(node, path, default=0.0):
            text = find_text(node, path)
            return float(text) if text is not None else default

        data = {
            'data_emissao': find_text(ide, 'nfe:dhEmi', '')[:10],
            'numero_nf': find_text(ide, 'nfe:nNF'),
            'cnpj_emitente': find_text(emit_info, 'nfe:CNPJ'),
            'nome_emitente': find_text(emit_info, 'nfe:xNome'),
            'uf_origem': find_text(emit, 'nfe:UF'),
            'cnpj_destinatario': find_text(dest_info, 'nfe:CNPJ'),
            'uf_destino': find_text(dest, 'nfe:UF'),
            
            # Valores totais da NF
            'valor_total_nf': find_float(total, 'nfe:vNF'),
            'valor_icms': find_float(total, 'nfe:vICMS'),
            'valor_icms_st': find_float(total, 'nfe:vICMSST'),
            'valor_fcp': find_float(total, 'nfe:vFCP'),
            'valor_difal': find_float(total, 'nfe:vICMSUFDest'),
        }
        return data
    except Exception as e:
        st.warning(f"Erro ao ler um arquivo XML: {e}")
        return None

# ==============================================================================
# MÓDULO 2: DETECÇÃO DE ERROS E INCONSISTÊNCIAS
# ==============================================================================
def validate_data_advanced(df):
    """Realiza validações avançadas, incluindo cruzamento de informações."""
    errors = []
    if df.empty:
        return df, pd.DataFrame(errors)

    df['status_validacao'] = 'OK'
    
    # --- LÓGICA DE CATEGORIZAÇÃO PARA AGENTE 2.0 ---
    # Contagem de erros de ICMS por fornecedor para análise de causa raiz
    df['icms_esperado'] = df['valor_total_nf'] * 0.18 # Alíquota simulada de 18%
    df['diferenca_icms'] = df['valor_icms'] - df['icms_esperado']
    icms_error_counts = df[abs(df['diferenca_icms']) > 1.00]['cnpj_emitente'].value_counts()


    # VALIDAÇÃO 1: Notas duplicadas
    duplicates = df[df.duplicated(subset=['numero_nf', 'cnpj_emitente'], keep=False)]
    for index, row in duplicates.iterrows():
        errors.append({
            'tipo': 'Duplicidade', 
            'detalhe': f"NF {row['numero_nf']} (CNPJ {row['cnpj_emitente']}) está duplicada.", 
            'valor_impacto': row['valor_total_nf'],
            'severidade': 'Crítico',
            'causa_provavel': 'Erro de processo no lançamento ou importação duplicada de arquivos.'
        })
        df.loc[index, 'status_validacao'] = 'ERRO'

    # VALIDAÇÃO 2: Valores zerados
    zero_value_notes = df[df['valor_total_nf'] <= 0]
    for index, row in zero_value_notes.iterrows():
        errors.append({
            'tipo': 'Valor Inválido', 
            'detalhe': f"NF {row['numero_nf']} tem valor zerado ou negativo.", 
            'valor_impacto': 0,
            'severidade': 'Aviso',
            'causa_provavel': 'Pode ser uma nota fiscal de estorno, devolução ou simples erro de digitação.'
        })
        df.loc[index, 'status_validacao'] = 'ERRO'
        
    # VALIDAÇÃO 3: Cruzamento de ICMS (SIMULAÇÃO)
    icms_divergente = df[abs(df['diferenca_icms']) > 1.00]
    for index, row in icms_divergente.iterrows():
        status = "pago a mais" if row['diferenca_icms'] > 0 else "pago a menos"
        causa_provavel = 'Cadastro de alíquota do produto/serviço incorreto.'
        if icms_error_counts.get(row['cnpj_emitente'], 0) > 1:
            causa_provavel = f'Fornecedor {row["nome_emitente"]} ({row["cnpj_emitente"]}) apresenta erros recorrentes. Verificar cadastro de alíquota deste fornecedor.'

        errors.append({
            'tipo': 'Divergência de ICMS', 
            'detalhe': f"NF {row['numero_nf']}: ICMS declarado (R${row['valor_icms']:.2f}) difere do esperado (R${row['icms_esperado']:.2f}). Possível ICMS {status}.", 
            'valor_impacto': row['diferenca_icms'],
            'severidade': 'Crítico',
            'causa_provavel': causa_provavel
        })
        df.loc[index, 'status_validacao'] = 'ERRO_ICMS'
        
    return df, pd.DataFrame(errors)

def generate_error_report_pdf(df_errors):
    """Gera um relatório PDF com a lista de inconsistências."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Relatorio de Inconsistencias Fiscais', 0, 1, 'C')
    pdf.ln(10)

    for index, row in df_errors.iterrows():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Tipo de Erro: {row['tipo']}", 0, 1)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 5, f"Detalhe: {row['detalhe']}")
        pdf.multi_cell(0, 5, f"Valor do Impacto (R$): {row['valor_impacto']:.2f}")
        # Adiciona os campos do Agente 2.0 no PDF também
        if 'severidade' in row:
             pdf.multi_cell(0, 5, f"Severidade: {row['severidade']}")
        if 'causa_provavel' in row:
             pdf.multi_cell(0, 5, f"Causa Provavel: {row['causa_provavel']}")
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin1')

# ==============================================================================
# MÓDULO 3: PLANEJAMENTO TRIBUTÁRIO
# ==============================================================================
def simulate_all_regimes(faturamento_anual):
    """Simula a carga tributária para Simples Nacional, Lucro Presumido e Lucro Real."""
    if faturamento_anual == 0:
        return {}
    
    # Lucro Presumido (Serviços)
    presumido_pis = faturamento_anual * 0.0065 # Alíquota PIS
    presumido_cofins = faturamento_anual * 0.03 # Alíquota COFINS
    base_irpj = faturamento_anual * 0.32 # Base de cálculo IRPJ para serviços
    base_csll = faturamento_anual * 0.32 # Base de cálculo CSLL para serviços
    presumido_irpj = base_irpj * 0.15 # Alíquota IRPJ
    presumido_csll = base_csll * 0.09 # Alíquota CSLL
    total_presumido = presumido_pis + presumido_cofins + presumido_irpj + presumido_csll

    # Simples Nacional (Anexo III - faturamento até 180k) - Simulação simplificada
    aliquota_simples = 0.06 # Alíquota inicial
    if faturamento_anual > 180000:
        aliquota_simples = 0.112 # Próxima faixa
    if faturamento_anual > 360000:
        aliquota_simples = 0.135
    total_simples = faturamento_anual * aliquota_simples

    # Lucro Real (Simulação com margem de lucro de 20%)
    lucro_apurado = faturamento_anual * 0.20 # Margem de lucro SIMULADA de 20%
    real_pis = faturamento_anual * 0.0165
    real_cofins = faturamento_anual * 0.076
    real_irpj = lucro_apurado * 0.15
    real_csll = lucro_apurado * 0.09
    total_real = real_pis + real_cofins + real_irpj + real_csll
    
    results = {
        'Lucro Presumido': total_presumido,
        'Simples Nacional (Simulado)': total_simples,
        'Lucro Real (Simulado)': total_real
    }
    return results

# ==============================================================================
# MÓDULO 4: PREVISÃO COM MACHINE LEARNING
# ==============================================================================
def predict_with_sklearn(df_monthly):
    df_monthly['time_idx'] = np.arange(len(df_monthly))
    X = df_monthly[['time_idx']]
    y = df_monthly['valor_total_nf']
    model = LinearRegression().fit(X, y)
    
    last_time_idx = df_monthly['time_idx'].max()
    future_time_idx = np.arange(last_time_idx + 1, last_time_idx + 4).reshape(-1, 1)
    future_predictions = model.predict(future_time_idx)
    
    last_date = df_monthly['data_emissao'].max()
    future_dates = pd.to_datetime([last_date + pd.DateOffset(months=i) for i in range(1, 4)])
    
    return pd.DataFrame({'Data': future_dates, 'Previsão': future_predictions})

def create_dataset_for_lstm(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def predict_with_tensorflow(df_monthly):
    from sklearn.preprocessing import MinMaxScaler
    dataset = df_monthly['valor_total_nf'].values.astype('float32').reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    look_back = 1
    trainX, trainY = create_dataset_for_lstm(dataset, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(4, input_shape=(1, look_back)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)

    last_val = dataset[-1]
    predictions = []
    for _ in range(3):
        pred_input = np.array([last_val]).reshape(1, 1, 1)
        prediction = model.predict(pred_input, verbose=0)
        predictions.append(prediction[0][0])
        last_val = prediction

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    last_date = df_monthly['data_emissao'].max()
    future_dates = pd.to_datetime([last_date + pd.DateOffset(months=i) for i in range(1, 4)])
    
    return pd.DataFrame({'Data': future_dates, 'Previsão': predictions.flatten()})

# ==============================================================================
# MÓDULO 5: COMPLIANCE E ALERTAS
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_rss_feed(feed_url='http://www.sped.fazenda.gov.br/spedtabelas/servicos/Depec/RSS.asmx/GetRSS'):
    try:
        feed = feedparser.parse(feed_url)
        return feed.entries
    except Exception as e:
        return [{'title': 'Erro ao buscar feed', 'summary': str(e), 'link': '#', 'published': ''}]

# ==============================================================================
# MÓDULO 6: ORQUESTRADOR DE AGENTES INTELIGENTES (OLLAMA)
# ==============================================================================
def invoke_agent(persona, contexto_tarefa):
    full_prompt = f"{persona}\n\n{contexto_tarefa}"
    try:
        response = ollama.chat(
            model='gemma:2b',
            messages=[{'role': 'user', 'content': full_prompt}]
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Erro ao contatar o agente Ollama. Verifique se ele está rodando. Detalhe: {e}")
        return None

# ==============================================================================
# FUNÇÃO AUXILIAR PARA DOWNLOAD
# ==============================================================================
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Relatorio')
    return output.getvalue()

# ==============================================================================
# INTERFACE PRINCIPAL DO STREAMLIT COM AGENTES
# ==============================================================================
st.title("🚀 Plataforma Fiscal com Agentes Inteligentes 2.0")

if 'audit_history' not in st.session_state:
    st.session_state.audit_history = []

with st.sidebar:
    st.header("1. Carregar Dados")
    uploaded_files = st.file_uploader("Selecione os arquivos XML de NF-e", type=['xml'], accept_multiple_files=True)
    st.info("Interaja com agentes especializados para obter insights profundos sobre seus dados fiscais.")
    st.warning("**Atenção:** Para usar os Agentes de IA, o Ollama com `gemma:2b` deve estar rodando localmente.")

if uploaded_files:
    with st.spinner('Lendo e processando os arquivos XML...'):
        all_data = [parse_nfe_xml(file) for file in uploaded_files]
        df = pd.DataFrame([d for d in all_data if d is not None])
        if df.empty:
            st.error("Nenhum dado válido pôde ser extraído dos arquivos XML.")
            st.stop()
        df['data_emissao'] = pd.to_datetime(df['data_emissao'])
    
    st.success(f"{len(df)} notas fiscais carregadas com sucesso!")

    # Normalizar para faturamento anual para simulação
    num_days = (df['data_emissao'].max() - df['data_emissao'].min()).days + 1
    total_revenue_periodo = df['valor_total_nf'].sum()
    faturamento_anual_projetado = (total_revenue_periodo / num_days) * 365 if num_days > 0 else 0

    df_validated, df_errors = validate_data_advanced(df.copy())
    regime_simulation_results = simulate_all_regimes(faturamento_anual_projetado)
    
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    audit_summary = {"timestamp": run_timestamp, "notas_processadas": len(df), "faturamento_total": f"R$ {total_revenue_periodo:,.2f}", "erros_encontrados": len(df_errors)}
    st.session_state.audit_history.insert(0, audit_summary)
    
    tab_list = ["📊 Dashboard", "🔍 Validação e Erros", "📈 Planejamento Tributário", "🔮 Previsão (ML)", "🏛️ Compliance e Alertas", "🤖 Assistente Geral"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_list)

    with tab1:
        st.header("Dashboard Geral")
        col1, col2, col3 = st.columns(3)
        col1.metric("Faturamento no Período", f"R$ {total_revenue_periodo:,.2f}")
        col2.metric("Total de Inconsistências", f"{len(df_errors)}")
        impacto_total = df_errors['valor_impacto'].sum() if not df_errors.empty else 0
        col3.metric("Impacto Financeiro (Erros)", f"R$ {impacto_total:,.2f}")
        st.dataframe(df_validated)
        st.download_button("📥 Baixar Relatório em Excel", to_excel(df_validated), "relatorio_fiscal.xlsx")

    with tab2:
        st.header("Relatório Detalhado de Inconsistências")
        if not df_errors.empty:
            st.warning("Foram encontrados os seguintes problemas nos dados:")
            st.dataframe(df_errors)
            
            st.download_button("📄 Baixar Relatório de Erros em PDF", generate_error_report_pdf(df_errors), "relatorio_inconsistencias.pdf", "application/pdf")
            
            st.markdown("---")
            if st.button("🤖 Chamar Agente Analista de Inconsistências"):
                persona_analista = "Você é um auditor fiscal sênior, extremamente meticuloso. Sua missão é analisar a lista de erros, explicar o impacto financeiro de cada um em termos simples e criar um plano de ação priorizado para corrigi-los."
                contexto_erros = f"Analise a seguinte lista de inconsistências fiscais e forneça um resumo executivo com um plano de ação claro. \n\nDados dos Erros:\n{df_errors.to_string()}"
                
                with st.spinner("O Agente Analista está investigando os dados... 🕵️‍♂️"):
                    resposta = invoke_agent(persona_analista, contexto_erros)
                    if resposta:
                        st.markdown("### Análise do Agente:")
                        st.info(resposta)
            
            # --- AGENTE 2.0: ANALISTA DE INCONSISTÊNCIAS COM ANÁLISE DE CAUSA RAIZ ---
            st.markdown("---")
            st.subheader("🕵️‍♂️ Agente Analista 2.0: Análise de Causa Raiz e Plano de Ação")
            if st.button("🤖 Chamar Agente Analista 2.0"):
                persona_analista_2_0 = """
                Você é um auditor fiscal sênior 2.0, especialista em diagnóstico e resolução de problemas. 
                Sua missão é analisar a lista de erros, que já contém uma categorização de severidade e uma causa provável, e transformá-la em um plano de ação claro e prático.
                - Para erros 'Críticos', defina ações imediatas.
                - Para erros de 'Aviso', sugira verificações.
                - Agrupe problemas semelhantes, como múltiplos erros do mesmo fornecedor.
                - Apresente o resultado como um checklist em formato Markdown (usando `- [ ]`).
                """
                df_errors_for_agent = df_errors[['tipo', 'detalhe', 'severidade', 'causa_provavel', 'valor_impacto']]
                contexto_erros_2_0 = f"Com base na tabela de erros a seguir, crie um plano de ação em formato de checklist priorizado por severidade. Agrupe as ações quando possível.\n\nDados dos Erros:\n{df_errors_for_agent.to_markdown()}"
                
                with st.spinner("O Agente Analista 2.0 está diagnosticando a causa raiz..."):
                    resposta = invoke_agent(persona_analista_2_0, contexto_erros_2_0)
                    if resposta:
                        st.markdown("### Plano de Ação Priorizado (Agente 2.0):")
                        st.info(resposta)

        else:
            st.success("Nenhuma inconsistência encontrada! ✅")

    with tab3:
        st.header("Simulação de Regimes Fiscais (Baseado em Faturamento Anual Projetado)")
        if regime_simulation_results:
            st.write(f"Faturamento anual projetado para simulação: **R$ {faturamento_anual_projetado:,.2f}**")
            df_regimes = pd.DataFrame(list(regime_simulation_results.items()), columns=['Regime', 'Carga Tributária (R$)'])
            fig = px.bar(df_regimes, x='Regime', y='Carga Tributária (R$)', text_auto='.2s', title="Comparativo de Carga Tributária Anual Simulada")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            if st.button("🤖 Chamar Agente Consultor Tributário"):
                persona_consultor = "Você é um consultor tributário estratégico. Seu objetivo é ajudar a empresa a economizar impostos de forma legal. Analise os dados do faturamento e da simulação para fornecer uma recomendação clara, considerando os prós, contras e riscos futuros."
                contexto_simulacao = f"Analise os seguintes dados e forneça uma recomendação de regime tributário para o próximo ano.\n- Faturamento Anual Projetado: R$ {faturamento_anual_projetado:,.2f}\n- Resultados da Simulação de Impostos: {regime_simulation_results}"
                with st.spinner("O Agente Consultor está elaborando a estratégia... 💡"):
                    resposta = invoke_agent(persona_consultor, contexto_simulacao)
                    if resposta:
                        st.markdown("### Recomendação Estratégica do Agente:")
                        st.info(resposta)
            
            # --- AGENTE 2.0: CONSULTOR TRIBUTÁRIO COM PLANEJAMENTO DE CENÁRIOS ---
            st.markdown("---")
            st.subheader("💡 Agente Consultor 2.0: Planejamento de Cenários Interativo")
            growth_percentage = st.slider("Qual sua expectativa de crescimento de faturamento para o próximo ano?", -20, 100, 15, format="%d%%")
            
            if st.button("🤖 Simular Cenário Futuro e Chamar Agente 2.0"):
                faturamento_futuro = faturamento_anual_projetado * (1 + growth_percentage / 100)
                simulacao_futura = simulate_all_regimes(faturamento_futuro)
                
                persona_consultor_2_0 = """
                Você é um consultor tributário estratégico 2.0, focado em planejamento de cenários. 
                Sua missão é analisar e comparar dois cenários: o atual e um futuro, com base na projeção de crescimento do usuário.
                1.  Compare a carga tributária em ambos os cenários.
                2.  Destaque se a recomendação de regime tributário muda com o crescimento.
                3.  Explique o "ponto de virada": qual fator (faturamento, lucro) faz um regime ser mais vantajoso que outro.
                4.  Forneça uma recomendação final considerando a projeção de crescimento.
                """
                contexto_simulacao_2_0 = f"""
                Analise os dois cenários a seguir e forneça uma recomendação estratégica.

                **Cenário Atual:**
                - Faturamento Anual Projetado: R$ {faturamento_anual_projetado:,.2f}
                - Simulação de Impostos Atual: {regime_simulation_results}

                **Cenário Futuro (Projeção do Usuário):**
                - Crescimento Esperado: {growth_percentage}%
                - Faturamento Futuro Projetado: R$ {faturamento_futuro:,.2f}
                - Simulação de Impostos Futura: {simulacao_futura}
                """
                with st.spinner("O Agente Consultor 2.0 está analisando os cenários..."):
                    resposta = invoke_agent(persona_consultor_2_0, contexto_simulacao_2_0)
                    if resposta:
                        st.markdown(f"### Análise de Cenários (Crescimento de {growth_percentage}%):")
                        
                        df_futuro = pd.DataFrame(list(simulacao_futura.items()), columns=['Regime', 'Carga Futura (R$)'])
                        df_comparativo = pd.merge(df_regimes, df_futuro, on='Regime')
                        st.dataframe(df_comparativo)
                        
                        st.info(resposta)

        else:
            st.warning("Faturamento zerado, não é possível simular.")
    
    with tab4:
        st.header("Previsão de Fluxo de Caixa (Faturamento)")
        df_monthly = df.set_index('data_emissao').resample('M')['valor_total_nf'].sum().reset_index()

        if len(df_monthly) < 3:
            st.warning("São necessários pelo menos 3 meses de dados para realizar uma previsão.")
        else:
            model_choice = st.selectbox("Escolha o modelo de previsão:", ["Regressão Linear (Scikit-Learn)", "Rede Neural LSTM (TensorFlow)"])
            if st.button("Gerar Previsão"):
                with st.spinner("Treinando modelo e gerando previsão..."):
                    if model_choice == "Regressão Linear (Scikit-Learn)":
                        forecast_df = predict_with_sklearn(df_monthly.copy())
                    else:
                        forecast_df = predict_with_tensorflow(df_monthly.copy())
                
                st.subheader(f"Previsão para os próximos 3 meses ({model_choice})")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_monthly['data_emissao'], y=df_monthly['valor_total_nf'], mode='lines+markers', name='Histórico Mensal'))
                fig.add_trace(go.Scatter(x=forecast_df['Data'], y=forecast_df['Previsão'], mode='lines+markers', name='Previsão', line=dict(dash='dot')))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(forecast_df)

    with tab5:
        st.header("Monitoramento de Compliance e Alertas")
        st.subheader("Últimas Notícias do Portal SPED")
        feed_entries = fetch_rss_feed()
        for entry in feed_entries[:5]:
            st.markdown(f"**[{entry.title}]({entry.link})** ({entry.published if 'published' in entry else 'N/A'})")
        
        st.markdown("---")
        if st.button("🤖 Chamar Agente de Compliance"):
            persona_compliance = "Você é um agente de compliance proativo. Sua função é ler as manchetes de notícias fiscais e, com base no perfil da empresa, identificar quais notícias são potencialmente relevantes e por quê. Seja conciso e direto ao ponto."
            contexto_empresa = "O perfil da minha empresa é de prestação de serviços, com faturamento anual na faixa do Lucro Presumido."
            titulos_noticias = "\n- ".join([entry.title for entry in feed_entries[:5]])
            contexto_tarefa = f"Analise as seguintes manchetes e diga quais são mais relevantes para uma empresa com este perfil. Justifique.\n\nPerfil da Empresa:\n{contexto_empresa}\n\nManchetes Recentes:\n- {titulos_noticias}"
            
            with st.spinner("O Agente de Compliance está monitorando as mudanças... 📡"):
                resposta = invoke_agent(persona_compliance, contexto_tarefa)
                if resposta:
                    st.markdown("### Alertas do Agente de Compliance:")
                    st.info(resposta)

        # --- AGENTE 2.0: COMPLIANCE COM MONITORAMENTO PERSONALIZADO ---
        st.markdown("---")
        st.subheader("📡 Agente de Compliance 2.0: Monitoramento Personalizado")
        keywords_input = st.text_input("Digite palavras-chave de seu interesse (ex: nfs-e, icms, simples nacional, e-commerce)", "substituição tributária, nfs-e")
        
        if st.button("🤖 Filtrar Notícias e Chamar Agente 2.0"):
            keywords = [k.strip().lower() for k in keywords_input.split(',')]
            relevant_entries = []
            for entry in feed_entries:
                content_to_search = f"{entry.title.lower()} {entry.summary.lower() if 'summary' in entry else ''}"
                if any(re.search(r'\b' + re.escape(keyword) + r'\b', content_to_search) for keyword in keywords):
                    relevant_entries.append(entry)
            
            if not relevant_entries:
                st.warning(f"Nenhuma notícia encontrada com as palavras-chave: {', '.join(keywords)}")
            else:
                persona_compliance_2_0 = """
                Você é um agente de compliance 2.0, um especialista em filtrar e contextualizar informações fiscais.
                Sua tarefa é analisar uma lista de notícias *previamente filtradas* com base nas palavras-chave do cliente.
                Para cada notícia, explique de forma concisa e direta **por que ela é relevante** para o negócio do cliente, considerando suas palavras-chave de interesse.
                Seja prático e foque no possível impacto ou ação necessária.
                """
                
                noticias_filtradas_str = "\n\n".join([f"**Título:** {e.title}\n**Link:** {e.link}" for e in relevant_entries])
                
                contexto_tarefa_2_0 = f"""
                Analise as seguintes notícias, que foram filtradas por serem relevantes para o cliente. Explique o impacto de cada uma.

                **Palavras-chave de Interesse do Cliente:** {', '.join(keywords)}

                **Notícias Relevantes:**
                {noticias_filtradas_str}
                """
                with st.spinner("O Agente de Compliance 2.0 está analisando as notícias personalizadas..."):
                    resposta = invoke_agent(persona_compliance_2_0, contexto_tarefa_2_0)
                    if resposta:
                        st.markdown("### Análise Personalizada do Agente 2.0:")
                        st.info(resposta)

        st.subheader("Histórico de Verificações (Nesta Sessão)")
        st.dataframe(pd.DataFrame(st.session_state.audit_history))

    with tab6:
        st.header("Assistente Geral")
        st.markdown("Faça uma pergunta geral sobre os dados carregados.")
        user_question = st.text_input("Sua pergunta:", placeholder="Ex: Faça um resumo executivo da situação fiscal da empresa.")
        
        if st.button("Perguntar ao Assistente Geral"):
            if user_question:
                persona_geral = "Você é um assistente geral de análise de dados. Sua tarefa é responder à pergunta do usuário com base no contexto fornecido."
                impacto_total = df_errors['valor_impacto'].sum() if not df_errors.empty else 0
                contexto_geral = f"""
                Resumo da análise atual:
                - Período: {df['data_emissao'].min().strftime('%d/%m/%Y')} a {df['data_emissao'].max().strftime('%d/%m/%Y')}
                - Faturamento no período: R$ {total_revenue_periodo:,.2f}
                - Faturamento anual projetado: R$ {faturamento_anual_projetado:,.2f}
                - Total de inconsistências encontradas: {len(df_errors)}
                - Impacto financeiro das inconsistências: R$ {impacto_total:,.2f}
                - Simulação de imposto (Lucro Presumido): R$ {regime_simulation_results.get('Lucro Presumido', 0):,.2f}
                
                Pergunta do usuário: "{user_question}"
                """
                with st.spinner("O assistente está processando sua pergunta..."):
                    resposta = invoke_agent(persona_geral, contexto_geral)
                    if resposta:
                        st.markdown("### Resposta do Assistente:")
                        st.info(resposta)
            else:
                st.warning("Por favor, digite uma pergunta.")

else:
    st.info("Aguardando o upload de arquivos XML para iniciar a análise.")

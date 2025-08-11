
# 🚀 Sistema Fiscal Inteligente com Agentes Especializados 2.0

Plataforma interativa construída com **Streamlit** que realiza análise fiscal inteligente de **Notas Fiscais Eletrônicas (NF-e)**, detecta inconsistências, simula regimes tributários, prevê faturamento com **Machine Learning** e interage com **agentes especializados** via Ollama.

---

## 📌 Funcionalidades

- **Ingestão de Dados**
  - Upload de múltiplos arquivos XML de NF-e.
  - Extração automática de informações chave (CNPJ, data, valor, ICMS etc.).

- **Validação e Auditoria**
  - Detecção de notas duplicadas, valores inválidos e divergências de ICMS.
  - Categorização por severidade e causa provável.
  - Geração de relatório PDF de inconsistências.

- **Planejamento Tributário**
  - Simulação de carga tributária para:
    - Lucro Presumido
    - Simples Nacional
    - Lucro Real
  - Comparação gráfica e análise de cenários.

- **Previsão de Faturamento**
  - Modelos de previsão:
    - Regressão Linear (Scikit-Learn)
    - Rede Neural LSTM (TensorFlow)

- **Compliance e Alertas**
  - Integração com feed RSS do Portal SPED.
  - Filtragem personalizada por palavras-chave.
  - Análise contextual de impacto de notícias.

- **Agentes Inteligentes (Ollama)**
  - Agente Analista de Inconsistências 2.0.
  - Agente Consultor Tributário 2.0.
  - Agente de Compliance 2.0.
  - Assistente Geral para perguntas sobre os dados.

---

## 📦 Requisitos

### Python
- Versão **3.10 ou superior** recomendada.

### Dependências
Arquivo `requirements.txt`:

```txt
pandas
numpy
scikit-learn
tensorflow
streamlit
openpyxl
plotly
ollama
fpdf
feedparser
````

Instalar dependências:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuração do Ollama

Este sistema utiliza agentes locais via [Ollama](https://ollama.ai/).

1. Instale o Ollama no seu sistema.
2. Baixe o modelo `gemma:2b`:

   ```bash
   ollama pull gemma:2b
   ```
3. Inicie o Ollama:

   ```bash
   ollama serve
   ```

---

## ▶️ Executando a Aplicação

1. Salve o código principal como `app.py`.
2. Inicie o Streamlit:

   ```bash
   streamlit run app.py
   ```
3. Abra no navegador o endereço indicado (geralmente `http://localhost:8501`).

---

## 📂 Estrutura do Projeto

```
.
├── app.py                 # Código principal da aplicação
├── requirements.txt       # Lista de dependências
├── README.md              # Documentação do projeto
└── data/                  # (opcional) Pasta para arquivos XML de NF-e
```

---

## 🛠️ Como Usar

1. **Carregar Dados**

   * No painel lateral, faça upload de um ou mais arquivos XML de NF-e.

2. **Explorar Resultados**

   * Navegue pelas abas:

     * **📊 Dashboard**
     * **🔍 Validação e Erros**
     * **📈 Planejamento Tributário**
     * **🔮 Previsão (ML)**
     * **🏛️ Compliance e Alertas**
     * **🤖 Assistente Geral**

3. **Interagir com Agentes**

   * Utilize os botões das abas para acionar os agentes especializados.

---

## 📜 Licença

Projeto livre para uso e adaptação para fins educacionais e empresariais.


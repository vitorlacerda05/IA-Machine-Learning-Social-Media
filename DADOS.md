# Instruções para Dados

## 📁 Arquivos Necessários

Para executar este projeto, você precisa dos seguintes arquivos:

### **Dados de Treinamento**
- **Arquivo**: `df_social_data_train.pkl`
- **Tamanho**: ~8.5MB
- **Conteúdo**: Dados de treinamento com colunas:
  - `anon_id`: Identificador anonimizado
  - `content`: Conteúdo textual do post
  - `reactions`: Número de reações
  - `comments`: Número de comentários
  - `engagement`: Classificação (alto/baixo)

### **Dados de Teste**
- **Arquivo**: `df_social_data_test.csv`
- **Tamanho**: ~3.5MB
- **Conteúdo**: Dados de teste com colunas:
  - `anon_id`: Identificador anonimizado
  - `content`: Conteúdo textual do post

## 🚀 Como Obter os Dados

### **Opção 1: Dados Fornecidos pela Disciplina**
Os arquivos devem ser fornecidos pelos professores da disciplina de Inteligência Artificial.

### **Opção 2: Executar o Código**
1. Execute `python main.py` para análise completa
2. Execute `python predictionKNN.py` para gerar:
   - `modelo_knn_engajamento.pkl` (modelo treinado)
   - `df_kaggle_knn.csv` (predições)

## 📊 Estrutura Esperada

```
supervised-classification-of-social-networks/
├── df_social_data_train.pkl    # Dados de treinamento
├── df_social_data_test.csv     # Dados de teste
├── modelo_knn_engajamento.pkl  # Modelo treinado (gerado)
├── df_kaggle_knn.csv          # Predições (gerado)
└── ... outros arquivos
```

## ⚠️ Nota Importante

Os arquivos de dados não estão incluídos no repositório devido ao tamanho. Eles devem ser obtidos separadamente conforme as instruções da disciplina. 
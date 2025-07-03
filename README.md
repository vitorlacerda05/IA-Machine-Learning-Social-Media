# Machine learning para análise de engajamento em redes sociais

## 📋 Descrição

Projeto que implementa e compara **diferentes abordagens de Machine Learning** para análise de engajamento em redes sociais, demonstrando tanto **aprendizado supervisionado** quanto **não supervisionado**.

### 🎯 Objetivos

1. **Classificação Supervisionada**: Prever se uma postagem terá **alto** ou **baixo engajamento** baseado no conteúdo textual
2. **Clustering Não Supervisionado**: Descobrir grupos naturais de postagens similares baseado apenas no conteúdo textual

## 🏆 Resultados dos Experimentos

### 🎯 Classificação Supervisionada
**K-Nearest Neighbors (KNN)** demonstrou ser o melhor modelo para classificação supervisionada, superando o SVM que inicialmente apresentou melhor performance na validação cruzada.

**Por que KNN foi o Melhor?**
- **Robustez**: Menos propenso a overfitting
- **Similaridade semântica**: Funciona excepcionalmente bem com embeddings
- **Adaptabilidade**: Captura melhor padrões dinâmicos de redes sociais

### 🔍 Clustering Não Supervisionado
**K-Means** foi implementado para descobrir padrões ocultos nos dados, agrupando postagens por similaridade textual sem usar informações de engajamento durante o treinamento.

## 🔬 Abordagens de Machine Learning Implementadas

### Aprendizado Supervisionado
1. **Probabilístico**: Naive Bayes, Regressão Logística
2. **Simbólico**: Árvore de Decisão, Random Forest
3. **Conexionista**: Rede Neural Multicamadas
4. **Estatístico**: SVM, KNN ⭐ **MELHOR**, Gradient Boosting

### Aprendizado Não Supervisionado
1. **Clustering**: K-Means com 15 clusters

## 📁 Estrutura do Projeto

```
supervised-classification-of-social-networks/
├── main.py                                # Análise completa dos 4 paradigmas
├── predictionKNN.py                       # Pipeline de predição com KNN (Supervisionado)
├── predictionKmeans.py                    # Pipeline de clustering com K-Means (Não Supervisionado)
├── requirements.txt                       # Dependências
├── README.md                             # Este arquivo
├── DADOS.md                              # Instruções para dados
├── .gitignore                            # Arquivos ignorados
├── results/                              # Pasta com resultados
└── theory/                               # Pasta com material teórico
```

**📁 Arquivos de Dados (não incluídos no repo):**
- `df_social_data_train.pkl` - Dados de treinamento (~8.5MB)
- `df_social_data_test.csv` - Dados de teste (~3.5MB)
- `modelo_knn_engajamento.pkl` - Modelo KNN treinado (~120MB)
- `modelo_kmeans_engajamento.pkl` - Modelo K-Means treinado (~120MB)
- `df_kaggle_knn.csv` - Predições KNN (~100KB)
- `df_kaggle_kmeans.csv` - Clusters K-Means (~100KB)

> **Nota**: Veja `DADOS.md` para instruções sobre como obter os dados.

## 🚀 Como Usar

### 1. Instalação
```bash
pip install -r requirements.txt
```

### 2. Análise Comparativa de Algoritmos Supervisionados
```bash
python main.py
```
> **Resultado**: Comparação completa de 4 paradigmas de ML supervisionado

### 3. Classificação Supervisionada com KNN
```bash
python predictionKNN.py
```
> **Resultado**: Modelo treinado com labels de engajamento

**Saídas:**
- `modelo_knn_engajamento.pkl` - Modelo treinado
- `df_kaggle_knn.csv` - Predições no formato de submissão

### 4. Clustering Não Supervisionado com K-Means
```bash
python predictionKmeans.py
```
> **Resultado**: Agrupamento baseado apenas na similaridade textual

**Saídas:**
- `modelo_kmeans_engajamento.pkl` - Modelo treinado
- `df_kaggle_kmeans.csv` - Clusters no formato de submissão

## 📚 Referências e Fundamentos Teóricos

### **Machine Learning**
- [scikit-learn Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html)
- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)

### **Processamento de Linguagem Natural**
- [Sentence Transformers](https://huggingface.co/sentence-transformers)
- [Embeddings Semânticos](https://www.sbert.net/)

### **Conceitos Fundamentais**
- **Aprendizado Supervisionado**: Classificação com dados rotulados
- **Aprendizado Não Supervisionado**: Clustering e descoberta de padrões
- **Validação Cruzada**: Estratégias de avaliação robusta
- **Overfitting**: Problema de generalização em ML

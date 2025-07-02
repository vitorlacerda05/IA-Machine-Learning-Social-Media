# Classificação Supervisionada de Engajamento em Redes Sociais

## 📋 Descrição

Projeto que implementa e compara **quatro paradigmas de aprendizado de máquina** para classificar engajamento de postagens em redes sociais.

### 🎯 Objetivo

Prever se uma postagem terá **alto** ou **baixo engajamento** baseado apenas no conteúdo textual.

> **Restrição:** Durante a inferência, apenas o conteúdo textual pode ser usado.

## 🏆 Resultado Final

**K-Nearest Neighbors (KNN)** demonstrou ser o melhor modelo, superando o SVM que inicialmente apresentou melhor performance na validação cruzada.

### 🎯 Por que KNN foi o Melhor?

- **Robustez**: Menos propenso a overfitting
- **Similaridade semântica**: Funciona excepcionalmente bem com embeddings
- **Adaptabilidade**: Captura melhor padrões dinâmicos de redes sociais

## 🔬 Paradigmas Implementados

1. **Probabilístico**: Naive Bayes, Regressão Logística
2. **Simbólico**: Árvore de Decisão, Random Forest
3. **Conexionista**: Rede Neural Multicamadas
4. **Estatístico**: SVM, KNN ⭐ **MELHOR**, Gradient Boosting

## 📁 Estrutura do Projeto

```
supervised-classification-of-social-networks/
├── main.py                                # Análise completa dos 4 paradigmas
├── predictionKNN.py                       # Pipeline de predição com KNN
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
- `modelo_knn_engajamento.pkl` - Modelo treinado (~120MB)
- `df_kaggle_knn.csv` - Predições (~100KB)

> **Nota**: Veja `DADOS.md` para instruções sobre como obter os dados.

## 🚀 Como Usar

### 1. Instalação
```bash
pip install -r requirements.txt
```

### 2. Análise Completa dos Paradigmas
```bash
python main.py
```

### 3. Predição com KNN (Melhor Modelo)
```bash
python predictionKNN.py
```

**Saídas:**
- `modelo_knn_engajamento.pkl` - Modelo treinado
- `df_kaggle_knn.csv` - Predições no formato de submissão

## 📊 Formato do CSV de Saída

| ID  | Engagement |
| --- | ---------- |
| 0   | alto       |
| 1   | baixo      |
| 2   | alto       |
| ... | ...        |

## 🔧 Características Técnicas

### **Processamento de Texto**
- **Sentence Transformers**: Modelo "all-MiniLM-L6-v2"
- **384 dimensões** de embeddings
- **Normalização automática** dos textos

### **Modelo KNN Otimizado**
```python
KNeighborsClassifier(
    n_neighbors=3,      # 3 vizinhos mais próximos
    metric="cosine",    # Distância cosseno
    weights="uniform"   # Peso uniforme
)
```

### **Validação**
- **Stratified Split**: 70% treino, 30% teste
- **Cross-Validation**: 5-fold
- **Métricas**: Acurácia, Precisão, Recall, F1-Score

## 🏆 Por que KNN Superou o SVM?

### **Overfitting do SVM**
- Kernel RBF muito complexo
- Sensível a outliers
- Pode memorizar dados de treinamento

### **Robustez do KNN**
- Baseado em similaridade semântica
- Menos propenso a overfitting
- Adapta-se dinamicamente aos novos dados

## 📚 Referências

- [scikit-learn Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- Sentence Transformers: [Hugging Face](https://huggingface.co/sentence-transformers)

## 👨‍💻 Autor

**Vitor Antonio de Almeida Lacerda** - NUSP: 12544761  
**Disciplina**: Inteligência Artificial

---

**Nota**: Projeto desenvolvido para a Atividade 1 de IA, focando na comparação dos quatro paradigmas de aprendizado de máquina. O KNN demonstrou ser o modelo mais eficaz para classificação de engajamento em redes sociais. 
"""
Atividade 1 - Classificação Supervisionada de Engajamento em Redes Sociais
Comparação dos quatro paradigmas de aprendizado de máquina:
1. Probabilístico (Naive Bayes, Regressão Logística)
2. Simbólico (Árvores de Decisão, Random Forest)
3. Conexionista (Redes Neurais)
4. Estatístico (SVM, KNN, Gradient Boosting)

Autor: Vitor Antonio de Almeida Lacerda
NUSP: 12544761
Disciplina: Inteligência Artificial
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('default')
sns.set_palette("husl")

class ClassificadorEngajamento:
    """
    Classe principal para classificação de engajamento em redes sociais
    implementando os quatro paradigmas de aprendizado de máquina
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.sentence_model = None
        self.label_encoder = LabelEncoder()
        self.modelos = {}
        self.resultados = {}
        
    def carregar_dados(self, arquivo='df_social_data_train.pkl'):
        """
        Carrega e prepara os dados de treinamento
        """
        print("=" * 60)
        print("CARREGANDO E PREPARANDO OS DADOS")
        print("=" * 60)
        
        # Carregar dados
        self.df = pd.read_pickle(arquivo)
        print(f"Dados carregados: {self.df.shape}")
        print(f"Colunas disponíveis: {self.df.columns.tolist()}")
        
        # Verificar dados
        print(f"\nPrimeiras linhas:")
        print(self.df.head())
        
        # Verificar valores únicos de engajamento
        print(f"\nValores únicos de engajamento: {self.df['engagement'].unique()}")
        print(f"Distribuição de engajamento:")
        print(self.df['engagement'].value_counts())
        
        # Remover valores nulos
        self.df = self.df.dropna()
        print(f"\nDados após remoção de nulos: {self.df.shape}")
        
        return self.df
    
    def preprocessar_texto(self):
        """
        Pré-processamento do texto usando Sentence Transformers
        """
        print("\n" + "=" * 60)
        print("PRÉ-PROCESSAMENTO DE TEXTO")
        print("=" * 60)
        
        # Carregar modelo Sentence Transformer
        print("Carregando modelo Sentence Transformer...")
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Extrair features usando embeddings
        print("Extraindo embeddings dos textos...")
        self.df['features'] = list(self.sentence_model.encode(
            self.df['content'].tolist(), 
            show_progress_bar=True
        ))
        
        print(f"Features extraídas: {len(self.df['features'][0])} dimensões")
        
        return self.df
    
    def preparar_dados_treinamento(self, test_size=0.3, random_state=42):
        """
        Prepara os dados para treinamento e teste
        """
        print("\n" + "=" * 60)
        print("PREPARANDO DADOS PARA TREINAMENTO")
        print("=" * 60)
        
        # Dividir dados em treino e teste
        df_train, df_test = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.df['engagement']
        )
        
        # Preparar features e labels
        X_train = df_train['features'].tolist()
        X_test = df_test['features'].tolist()
        
        # Codificar labels
        y_train = self.label_encoder.fit_transform(df_train['engagement'])
        y_test = self.label_encoder.transform(df_test['engagement'])
        
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Conjunto de treino: {self.X_train.shape}")
        print(f"Conjunto de teste: {self.X_test.shape}")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Distribuição de classes (treino): {np.bincount(self.y_train)}")
        print(f"Distribuição de classes (teste): {np.bincount(self.y_test)}")
        
        return df_train, df_test
    
    def paradigma_probabilistico(self):
        """
        Implementação do paradigma probabilístico
        """
        print("\n" + "=" * 60)
        print("PARADIGMA PROBABILÍSTICO")
        print("=" * 60)
        
        from sklearn.naive_bayes import MultinomialNB, GaussianNB
        from sklearn.linear_model import LogisticRegression
        
        # 1. Naive Bayes (Gaussian para features contínuas)
        print("1. Treinando Naive Bayes...")
        nb_model = GaussianNB()
        nb_model.fit(self.X_train, self.y_train)
        nb_pred = nb_model.predict(self.X_test)
        nb_accuracy = accuracy_score(self.y_test, nb_pred)
        
        # Validação cruzada
        nb_cv_scores = cross_val_score(nb_model, self.X_train, self.y_train, cv=5)
        
        print(f"   Acurácia Naive Bayes: {nb_accuracy:.4f}")
        print(f"   CV Score: {nb_cv_scores.mean():.4f} (+/- {nb_cv_scores.std() * 2:.4f})")
        
        # 2. Regressão Logística
        print("2. Treinando Regressão Logística...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.X_train, self.y_train)
        lr_pred = lr_model.predict(self.X_test)
        lr_accuracy = accuracy_score(self.y_test, lr_pred)
        
        # Validação cruzada
        lr_cv_scores = cross_val_score(lr_model, self.X_train, self.y_train, cv=5)
        
        print(f"   Acurácia Regressão Logística: {lr_accuracy:.4f}")
        print(f"   CV Score: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std() * 2:.4f})")
        
        # Armazenar resultados
        self.modelos['Naive_Bayes'] = nb_model
        self.modelos['Regressao_Logistica'] = lr_model
        self.resultados['Naive_Bayes'] = nb_accuracy
        self.resultados['Regressao_Logistica'] = lr_accuracy
        
        return nb_model, lr_model, nb_accuracy, lr_accuracy
    
    def paradigma_simbolico(self):
        """
        Implementação do paradigma simbólico
        """
        print("\n" + "=" * 60)
        print("PARADIGMA SIMBÓLICO")
        print("=" * 60)
        
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        
        # 1. Árvore de Decisão
        print("1. Treinando Árvore de Decisão...")
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
        dt_model.fit(self.X_train, self.y_train)
        dt_pred = dt_model.predict(self.X_test)
        dt_accuracy = accuracy_score(self.y_test, dt_pred)
        
        # Validação cruzada
        dt_cv_scores = cross_val_score(dt_model, self.X_train, self.y_train, cv=5)
        
        print(f"   Acurácia Árvore de Decisão: {dt_accuracy:.4f}")
        print(f"   CV Score: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std() * 2:.4f})")
        
        # 2. Random Forest
        print("2. Treinando Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(self.X_train, self.y_train)
        rf_pred = rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)
        
        # Validação cruzada
        rf_cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, cv=5)
        
        print(f"   Acurácia Random Forest: {rf_accuracy:.4f}")
        print(f"   CV Score: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")
         
        # Armazenar resultados
        self.modelos['Arvore_Decisao'] = dt_model
        self.modelos['Random_Forest'] = rf_model
        self.resultados['Arvore_Decisao'] = dt_accuracy
        self.resultados['Random_Forest'] = rf_accuracy
        
        return dt_model, rf_model, dt_accuracy, rf_accuracy
    
    def paradigma_conexionista(self):
        """
        Implementação do paradigma conexionista (redes neurais)
        """
        print("\n" + "=" * 60)
        print("PARADIGMA CONEXIONISTA")
        print("=" * 60)
        
        from sklearn.neural_network import MLPClassifier
        
        # Rede Neural Multicamadas
        print("Treinando Rede Neural Multicamadas...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        nn_model.fit(self.X_train, self.y_train)
        nn_pred = nn_model.predict(self.X_test)
        nn_accuracy = accuracy_score(self.y_test, nn_pred)
        
        # Validação cruzada
        nn_cv_scores = cross_val_score(nn_model, self.X_train, self.y_train, cv=5)
        
        print(f"   Acurácia Rede Neural: {nn_accuracy:.4f}")
        print(f"   CV Score: {nn_cv_scores.mean():.4f} (+/- {nn_cv_scores.std() * 2:.4f})")
        
        # Armazenar resultados
        self.modelos['Rede_Neural'] = nn_model
        self.resultados['Rede_Neural'] = nn_accuracy
        
        return nn_model, nn_accuracy
    
    def paradigma_estatistico(self):
        """
        Implementação do paradigma estatístico
        """
        print("\n" + "=" * 60)
        print("PARADIGMA ESTATÍSTICO")
        print("=" * 60)
        
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        
        # 1. Support Vector Machine
        print("1. Treinando SVM...")
        svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        svm_model.fit(self.X_train, self.y_train)
        svm_pred = svm_model.predict(self.X_test)
        svm_accuracy = accuracy_score(self.y_test, svm_pred)
        
        # Validação cruzada
        svm_cv_scores = cross_val_score(svm_model, self.X_train, self.y_train, cv=5)
        
        print(f"   Acurácia SVM: {svm_accuracy:.4f}")
        print(f"   CV Score: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std() * 2:.4f})")
        
        # 2. K-Nearest Neighbors (otimizado)
        print("2. Treinando KNN com otimização de parâmetros...")
        
        # Parâmetros para GridSearch
        knn_params = {
            'n_neighbors': [1, 3, 5, 7, 9],
            'metric': ['cosine', 'euclidean'],
            'weights': ['uniform', 'distance']
        }
        
        knn_base = KNeighborsClassifier()
        knn_grid = GridSearchCV(knn_base, knn_params, cv=5, scoring='accuracy', n_jobs=-1)
        knn_grid.fit(self.X_train, self.y_train)
        
        knn_model = knn_grid.best_estimator_
        knn_pred = knn_model.predict(self.X_test)
        knn_accuracy = accuracy_score(self.y_test, knn_pred)
        
        print(f"   Melhores parâmetros KNN: {knn_grid.best_params_}")
        print(f"   Acurácia KNN: {knn_accuracy:.4f}")
        print(f"   CV Score: {knn_grid.best_score_:.4f}")
        
        # 3. Gradient Boosting
        print("3. Treinando Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(self.X_train, self.y_train)
        gb_pred = gb_model.predict(self.X_test)
        gb_accuracy = accuracy_score(self.y_test, gb_pred)
        
        # Validação cruzada
        gb_cv_scores = cross_val_score(gb_model, self.X_train, self.y_train, cv=5)
        
        print(f"   Acurácia Gradient Boosting: {gb_accuracy:.4f}")
        print(f"   CV Score: {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std() * 2:.4f})")
        
        # Armazenar resultados
        self.modelos['SVM'] = svm_model
        self.modelos['KNN'] = knn_model
        self.modelos['Gradient_Boosting'] = gb_model
        self.resultados['SVM'] = svm_accuracy
        self.resultados['KNN'] = knn_accuracy
        self.resultados['Gradient_Boosting'] = gb_accuracy
        
        return svm_model, knn_model, gb_model, svm_accuracy, knn_accuracy, gb_accuracy
    
    def comparar_resultados(self):
        """
        Compara e visualiza os resultados de todos os paradigmas
        """
        print("\n" + "=" * 60)
        print("COMPARAÇÃO DOS RESULTADOS")
        print("=" * 60)
        
        # Ordenar resultados por acurácia
        resultados_ordenados = sorted(self.resultados.items(), key=lambda x: x[1], reverse=True)
        
        print("\nRanking de Performance:")
        print("-" * 40)
        for i, (modelo, acuracia) in enumerate(resultados_ordenados, 1):
            print(f"{i:2d}. {modelo:20}: {acuracia:.4f}")
        
        # Melhor modelo
        melhor_modelo, melhor_acuracia = resultados_ordenados[0]
        print(f"\n🏆 MELHOR MODELO: {melhor_modelo} com acurácia de {melhor_acuracia:.4f}")
        
        # Visualização
        self.visualizar_resultados()
        
        return melhor_modelo, melhor_acuracia
    
    def visualizar_resultados(self):
        """
        Cria visualizações dos resultados
        """
        # Preparar dados para visualização
        modelos = list(self.resultados.keys())
        acuracias = list(self.resultados.values())
        
        # Criar figura com subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Gráfico de barras
        bars = ax1.bar(modelos, acuracias, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange', 'lightblue'])
        ax1.set_title('Comparação de Acurácia por Paradigma', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Acurácia')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Adicionar valores nas barras
        for bar, acc in zip(bars, acuracias):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Gráfico de pizza
        ax2.pie(acuracias, labels=modelos, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribuição de Performance', fontsize=14, fontweight='bold')
        
        # 3. Heatmap de comparação
        # Agrupar por paradigma
        paradigmas = {
            'Probabilístico': ['Naive_Bayes', 'Regressao_Logistica'],
            'Simbólico': ['Arvore_Decisao', 'Random_Forest'],
            'Conexionista': ['Rede_Neural'],
            'Estatístico': ['SVM', 'KNN', 'Gradient_Boosting']
        }
        
        paradigma_acuracia = {}
        for paradigma, modelos_paradigma in paradigmas.items():
            acuracias_paradigma = [self.resultados[modelo] for modelo in modelos_paradigma if modelo in self.resultados]
            paradigma_acuracia[paradigma] = np.mean(acuracias_paradigma) if acuracias_paradigma else 0
        
        paradigmas_list = list(paradigma_acuracia.keys())
        acuracias_paradigmas = list(paradigma_acuracia.values())
        
        bars3 = ax3.bar(paradigmas_list, acuracias_paradigmas, color=['red', 'blue', 'green', 'orange'])
        ax3.set_title('Performance Média por Paradigma', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Acurácia Média')
        ax3.set_ylim(0, 1)
        
        for bar, acc in zip(bars3, acuracias_paradigmas):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Boxplot de validação cruzada (exemplo com alguns modelos)
        cv_data = []
        cv_labels = []
        
        # Coletar dados de CV para alguns modelos
        modelos_cv = ['Naive_Bayes', 'Random_Forest', 'Rede_Neural', 'KNN']
        for modelo in modelos_cv:
            if modelo in self.modelos:
                cv_scores = cross_val_score(self.modelos[modelo], self.X_train, self.y_train, cv=5)
                cv_data.append(cv_scores)
                cv_labels.append(modelo)
        
        if cv_data:
            ax4.boxplot(cv_data, labels=cv_labels)
            ax4.set_title('Distribuição de Scores - Validação Cruzada', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Acurácia')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def analisar_melhor_modelo(self, nome_modelo):
        """
        Análise detalhada do melhor modelo
        """
        print(f"\n" + "=" * 60)
        print(f"ANÁLISE DETALHADA DO MELHOR MODELO: {nome_modelo}")
        print("=" * 60)
        
        modelo = self.modelos[nome_modelo]
        y_pred = modelo.predict(self.X_test)
        
        # Relatório de classificação
        print("\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
        print("-" * 40)
        print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Matriz de confusão
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_, 
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Matriz de Confusão - {nome_modelo}', fontsize=14, fontweight='bold')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predito')
        plt.show()
        
        # Análise de features importantes (se aplicável)
        if hasattr(modelo, 'feature_importances_'):
            print(f"\n🔍 TOP 10 FEATURES MAIS IMPORTANTES:")
            print("-" * 40)
            importancias = modelo.feature_importances_
            indices = np.argsort(importancias)[::-1]
            
            for i in range(min(10, len(importancias))):
                print(f"{i+1:2d}. Feature {indices[i]:4d}: {importancias[indices[i]]:.4f}")
        
        elif hasattr(modelo, 'coef_'):
            print(f"\n🔍 TOP 10 COEFICIENTES MAIS IMPORTANTES:")
            print("-" * 40)
            coeficientes = np.abs(modelo.coef_[0])
            indices = np.argsort(coeficientes)[::-1]
            
            for i in range(min(10, len(coeficientes))):
                print(f"{i+1:2d}. Feature {indices[i]:4d}: {coeficientes[indices[i]]:.4f}")
    
    def executar_analise_completa(self):
        """
        Executa a análise completa dos quatro paradigmas
        """
        print("🚀 INICIANDO ANÁLISE COMPLETA DOS QUATRO PARADIGMAS")
        print("=" * 70)
        
        # 1. Carregar e preparar dados
        self.carregar_dados()
        self.preprocessar_texto()
        df_train, df_test = self.preparar_dados_treinamento()
        
        # 2. Executar todos os paradigmas
        print("\n" + "🔬 EXECUTANDO OS QUATRO PARADIGMAS")
        print("=" * 50)
        
        # Paradigma Probabilístico
        self.paradigma_probabilistico()
        
        # Paradigma Simbólico
        self.paradigma_simbolico()
        
        # Paradigma Conexionista
        self.paradigma_conexionista()
        
        # Paradigma Estatístico
        self.paradigma_estatistico()
        
        # 3. Comparar resultados
        melhor_modelo, melhor_acuracia = self.comparar_resultados()
        
        # 4. Análise detalhada do melhor modelo
        self.analisar_melhor_modelo(melhor_modelo)
        
        # 5. Resumo final
        print(f"\n" + "=" * 60)
        print("📋 RESUMO FINAL")
        print("=" * 60)
        print(f"✅ Análise concluída com sucesso!")
        print(f"🏆 Melhor paradigma: {melhor_modelo}")
        print(f"📈 Melhor acurácia: {melhor_acuracia:.4f}")
        print(f"🔢 Total de modelos testados: {len(self.modelos)}")
        print(f"📊 Paradigmas implementados: Probabilístico, Simbólico, Conexionista, Estatístico")
        
        return self.modelos, self.resultados

# Função principal
def main():
    """
    Função principal que executa toda a análise
    """
    # Criar instância do classificador
    classificador = ClassificadorEngajamento()
    
    # Executar análise completa
    modelos, resultados = classificador.executar_analise_completa()
    
    return classificador, modelos, resultados

if __name__ == "__main__":
    # Executar análise
    classificador, modelos, resultados = main()

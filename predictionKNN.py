import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class PreditorEngajamentoKNN:
    """
    Classe para treinar o modelo KNN e fazer predições no conjunto de teste
    """
    
    def __init__(self):
        self.sentence_model = None
        self.knn_model = None
        self.label_encoder = LabelEncoder()
        
    def carregar_dados_treino(self, arquivo='df_social_data_train.pkl'):
        """
        PASSO 1 - Carrega e prepara os dados de treinamento
        """
        print("=" * 60)
        print("PASSO 1 - CARREGANDO DADOS DE TREINAMENTO")
        print("=" * 60)
        
        # Carregar dados de treinamento
        df_train = pd.read_pickle(arquivo)
        print(f"Dados de treinamento carregados: {df_train.shape}")
        print(f"Colunas: {df_train.columns.tolist()}")
        
        # Verificar dados
        print(f"\nPrimeiras linhas:")
        print(df_train.head())
        
        # Verificar valores únicos de engajamento
        print(f"\nValores únicos de engajamento: {df_train['engagement'].unique()}")
        print(f"Distribuição de engajamento:")
        print(df_train['engagement'].value_counts())
        
        # Remover valores nulos
        df_train = df_train.dropna()
        print(f"\nDados após remoção de nulos: {df_train.shape}")
        
        return df_train
    
    def preprocessar_texto_treino(self, df_train):
        """
        Pré-processamento do texto usando Sentence Transformers
        """
        print("\n" + "=" * 60)
        print("PRÉ-PROCESSAMENTO DE TEXTO - TREINAMENTO")
        print("=" * 60)
        
        # Carregar modelo Sentence Transformer
        print("Carregando modelo Sentence Transformer...")
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Extrair features usando embeddings
        print("Extraindo embeddings dos textos de treinamento...")
        df_train['features'] = list(self.sentence_model.encode(
            df_train['content'].tolist(), 
            show_progress_bar=True
        ))
        
        print(f"Features extraídas: {len(df_train['features'][0])} dimensões")
        
        return df_train
    
    def treinar_modelo_knn(self, df_train):
        """
        Treina o modelo KNN
        """
        print("\n" + "=" * 60)
        print("TREINANDO MODELO KNN")
        print("=" * 60)
        
        # Preparar features e labels
        X_train = np.array(df_train['features'].tolist())
        
        # Codificar labels
        y_train = self.label_encoder.fit_transform(df_train['engagement'])
        
        print(f"Features de treinamento: {X_train.shape}")
        print(f"Labels de treinamento: {y_train.shape}")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Distribuição de classes: {np.bincount(y_train)}")
        
        # Treinar modelo KNN
        print("\nTreinando KNN...")
        self.knn_model = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        self.knn_model.fit(X_train, y_train)
        
        print("✅ Modelo KNN treinado com sucesso!")
        print(f"Parâmetros do modelo: {self.knn_model}")
        
        return self.knn_model
    
    def salvar_modelo(self, nome_arquivo='modelo_knn_engajamento.pkl'):
        """
        Salva o modelo treinado e componentes necessários
        """
        print("\n" + "=" * 60)
        print("SALVANDO MODELO")
        print("=" * 60)
        
        # Criar dicionário com todos os componentes necessários
        modelo_completo = {
            'sentence_model': self.sentence_model,
            'knn_model': self.knn_model,
            'label_encoder': self.label_encoder
        }
        
        # Salvar modelo
        joblib.dump(modelo_completo, nome_arquivo)
        print(f"✅ Modelo salvo em: {nome_arquivo}")
        
        return nome_arquivo
    
    def carregar_dados_teste(self, arquivo='df_social_data_test.csv'):
        """
        PASSO 2 - Carrega os dados de teste
        """
        print("\n" + "=" * 60)
        print("PASSO 2 - CARREGANDO DADOS DE TESTE")
        print("=" * 60)
        
        # Carregar dados de teste
        df_test = pd.read_csv(arquivo)
        print(f"Dados de teste carregados: {df_test.shape}")
        print(f"Colunas: {df_test.columns.tolist()}")
        
        # Verificar dados
        print(f"\nPrimeiras linhas:")
        print(df_test.head())
        
        # Converter conteúdo para string (caso não seja)
        df_test['content'] = df_test['content'].astype(str)
        
        return df_test
    
    def preprocessar_texto_teste(self, df_test):
        """
        Pré-processamento do texto de teste
        """
        print("\n" + "=" * 60)
        print("PRÉ-PROCESSAMENTO DE TEXTO - TESTE")
        print("=" * 60)
        
        # Extrair features usando o mesmo modelo
        print("Extraindo embeddings dos textos de teste...")
        df_test['features'] = list(self.sentence_model.encode(
            df_test['content'].tolist(), 
            show_progress_bar=True
        ))
        
        print(f"Features extraídas: {len(df_test['features'][0])} dimensões")
        
        return df_test
    
    def fazer_predicoes(self, df_test):
        """
        Faz as predições no conjunto de teste
        """
        print("\n" + "=" * 60)
        print("FAZENDO PREDIÇÕES")
        print("=" * 60)
        
        # Preparar features de teste
        X_test = np.array(df_test['features'].tolist())
        print(f"Features de teste: {X_test.shape}")
        
        # Fazer predições
        print("Fazendo predições com o modelo KNN...")
        y_pred_encoded = self.knn_model.predict(X_test)
        
        # Decodificar predições
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Adicionar predições ao dataframe
        df_test['Engagement'] = y_pred_decoded
        
        print(f"✅ Predições concluídas!")
        print(f"Distribuição das predições:")
        print(df_test['Engagement'].value_counts())
        
        return df_test
    
    def gerar_csv_saida(self, df_test, nome_arquivo='df_kaggle_knn.csv'):
        """
        Gera o CSV de saída no formato especificado
        """
        print("\n" + "=" * 60)
        print("GERANDO CSV DE SAÍDA")
        print("=" * 60)
        
        # Criar coluna ID sequencial
        df_test['ID'] = range(len(df_test))
        
        # Selecionar apenas as colunas necessárias
        df_saida = df_test[['ID', 'Engagement']]
        
        # Salvar CSV
        df_saida.to_csv(nome_arquivo, index=False)
        
        print(f"✅ CSV gerado com sucesso: {nome_arquivo}")
        print(f"Formato: {df_saida.shape[0]} linhas, {df_saida.shape[1]} colunas")
        
        # Mostrar primeiras linhas
        print(f"\nPrimeiras linhas do CSV gerado:")
        print(df_saida.head(10))
        
        # Verificar formato
        print(f"\nVerificações do CSV:")
        print(f"✅ IDs únicos e sequenciais: {len(df_saida['ID'].unique()) == len(df_saida)}")
        print(f"✅ Valores de Engagement únicos: {df_saida['Engagement'].unique()}")
        print(f"✅ Sem índice no CSV: index=False")
        
        return df_saida
    
    def executar_pipeline_completo(self):
        """
        Executa o pipeline completo de treinamento e predição
        """
        print("🚀 INICIANDO PIPELINE DE PREDIÇÃO - KNN")
        print("=" * 70)
        
        try:
            # PASSO 1 - Treinar e salvar o modelo KNN
            print("\n" + "🔧 PASSO 1: TREINAMENTO E SALVAMENTO DO MODELO KNN")
            print("=" * 50)
            
            # Carregar dados de treinamento
            df_train = self.carregar_dados_treino()
            
            # Pré-processar texto
            df_train = self.preprocessar_texto_treino(df_train)
            
            # Treinar modelo KNN
            self.treinar_modelo_knn(df_train)
            
            # Salvar modelo
            self.salvar_modelo()
            
            # PASSO 2 - Fazer predições no conjunto de teste
            print("\n" + "🎯 PASSO 2: PREDIÇÕES NO CONJUNTO DE TESTE")
            print("=" * 50)
            
            # Carregar dados de teste
            df_test = self.carregar_dados_teste()
            
            # Pré-processar texto de teste
            df_test = self.preprocessar_texto_teste(df_test)
            
            # Fazer predições
            df_test = self.fazer_predicoes(df_test)
            
            # Gerar CSV de saída
            df_saida = self.gerar_csv_saida(df_test)
            
            # Resumo final
            print(f"\n" + "=" * 60)
            print("📋 RESUMO FINAL")
            print("=" * 60)
            print(f"✅ Pipeline executado com sucesso!")
            print(f"🏆 Modelo utilizado: KNN")
            print(f"📊 Dados de treinamento: {df_train.shape}")
            print(f"📊 Dados de teste: {df_test.shape}")
            print(f"📄 CSV gerado: df_kaggle_knn.csv")
            print(f"🔢 Total de predições: {len(df_saida)}")
            
            return df_saida
            
        except Exception as e:
            print(f"❌ Erro durante a execução: {e}")
            raise

# Função principal
def main():
    """
    Função principal que executa o pipeline completo
    """
    # Criar instância do preditor
    preditor = PreditorEngajamentoKNN()
    
    # Executar pipeline completo
    df_saida = preditor.executar_pipeline_completo()
    
    return preditor, df_saida

if __name__ == "__main__":
    # Executar pipeline
    preditor, df_saida = main() 
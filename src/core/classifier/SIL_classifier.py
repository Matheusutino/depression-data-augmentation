import optuna
from optuna.samplers import TPESampler
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
import random

class SILDataClassifier:
    def __init__(self, embed_name, random_seed=42, device="cuda"):
        """Inicializa a classe SIL com o nome do modelo de embedding e uma seed"""
        self.embed_name = embed_name
        self.model = SentenceTransformer(embed_name)
        self.scaler = StandardScaler()
        self.random_seed = random_seed
        self.device = device

    def generate_embeddings(self, posts):
        """Gera embeddings para os posts de um usuário usando o modelo especificado"""
        return self.model.encode(posts)

    def prepare_data(self, X, y):
        """Prepara os dados para o formato SIL"""
        X_flattened = [post for sublist in X for post in sublist]  # Flatten X
        y_flattened = [label for label, posts in zip(y, X) for _ in posts]  # Replicar as labels para cada post
        return X_flattened, y_flattened

    def objective(self, trial, x_train_pool, y_train, x_val_pool, y_val):
        """Função objetivo para o Optuna"""
        
        # Definir os parâmetros do modelo a serem otimizados
        eta = trial.suggest_loguniform('eta', 0.01, 0.3)  
        gamma = trial.suggest_uniform('gamma', 0, 0.2)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        subsample = trial.suggest_uniform('subsample', 0.8, 1.0)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 5)
        lambda_ = trial.suggest_uniform('lambda', 0, 10.0)
        alpha = trial.suggest_uniform('alpha', 0, 10.0)
        
        # Parâmetros do modelo XGBoost
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': eta,
            'gamma': gamma,
            'max_depth': max_depth,
            'subsample': subsample,
            'min_child_weight': min_child_weight,
            'lambda': lambda_,
            'alpha': alpha,
            'device': self.device,
            'predictor': 'gpu_predictor',
            'seed': self.random_seed
        }
        
        # Treinar o modelo com os parâmetros definidos
        dtrain = xgb.DMatrix(x_train_pool, label=y_train)
        dval = xgb.DMatrix(x_val_pool, label=y_val)
        
        clf = xgb.train(params, dtrain, num_boost_round=1000, 
                        evals=[(dval, 'eval')],
                        early_stopping_rounds=50)
        
        # Avaliar no conjunto de validação
        y_val_pred = clf.predict(dval)
        y_val_pred_binary = (y_val_pred > 0.5).astype(int)
        
        # Calcular as métricas
        f1 = f1_score(y_val, y_val_pred_binary)
        
        return f1

    def optimize_hyperparameters(self, x_train_pool, y_train, x_val_pool, y_val, n_trials):
        """Função para otimizar hiperparâmetros usando Optuna e retornar o histórico completo"""
        
        # Criar estudo Optuna
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)  
        study.optimize(lambda trial: self.objective(trial, x_train_pool, y_train, x_val_pool, y_val), n_trials=n_trials)
        
        # Capturar histórico de otimização
        trials_history = [{
            "trial_number": t.number,
            "params": t.params,
            "f1-score": t.value
        } for t in study.trials]
        
        return study.best_params, trials_history

    def predict_majority_label(self, clf, X):
        """Previsão por classe majoritária para cada conjunto de posts"""
        y_pred_majority = []
        
        # Para cada conjunto de posts (cada instância)
        for posts in X:
            # Gerar embeddings para os posts (espera-se que 'posts' seja uma lista de strings)
            post_embeds = self.generate_embeddings(posts)
            
            # Normalizar os embeddings (garante que 'post_embeds' esteja no formato esperado)
            post_embeds_pool = self.scaler.transform(post_embeds)
            
            # Converter para DMatrix antes de passar para o modelo
            dmatrix = xgb.DMatrix(post_embeds_pool)
            
            # Fazer a previsão para cada post
            post_preds = clf.predict(dmatrix)
            
            # Converter as probabilidades para 0 ou 1
            post_labels = (post_preds > 0.5).astype(int)
            
            # Obter a label majoritária (classe mais frequente)
            majority_label = np.bincount(post_labels).argmax()
            
            y_pred_majority.append(majority_label)
        
        return y_pred_majority


    def train_and_evaluate(self, x_train, x_val, x_test, y_train, y_val, y_test, n_trials=50, **kwargs):
        """Treina o modelo XGBoost utilizando Optuna para otimização dos hiperparâmetros"""
        
        # Preparar os dados para SIL
        x_train_flattened, y_train_flattened = self.prepare_data(x_train, y_train)
        x_val_flattened, y_val_flattened = self.prepare_data(x_val, y_val)  
        
        # Gerar embeddings para todos os conjuntos de dados
        x_train_embeds = self.generate_embeddings(x_train_flattened)
        x_val_embeds = self.generate_embeddings(x_val_flattened)
        
        # Normalizar embeddings
        x_train_pool = self.scaler.fit_transform(x_train_embeds)
        x_val_pool = self.scaler.transform(x_val_embeds)
        
        # Otimizar os hiperparâmetros usando Optuna
        best_params, trials_history = self.optimize_hyperparameters(x_train_pool, y_train_flattened, x_val_pool, y_val_flattened, n_trials)
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'predictor': 'gpu_predictor',
            'device': self.device ,
            'seed': self.random_seed
        })

        # Treinar o modelo final com os melhores parâmetros
        dtrain = xgb.DMatrix(x_train_pool, label=y_train_flattened)
        dval = xgb.DMatrix(x_val_pool, label=y_val_flattened)
        
        final_clf = xgb.train(best_params, dtrain, num_boost_round=1000, 
                        evals=[(dval, 'eval')],
                        early_stopping_rounds=50)
        
        # Predição para o conjunto de teste usando a previsão majoritária
        y_test_pred_majority = self.predict_majority_label(final_clf, x_test)

        # Calcular as métricas para a previsão majoritária
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred_majority),
            "precision": precision_score(y_test, y_test_pred_majority),
            "recall": recall_score(y_test, y_test_pred_majority),
            "f1_score": f1_score(y_test, y_test_pred_majority)
        }
        
        result = {
            "best_params": best_params,
            "test_metrics": test_metrics
        }
        
        return result, trials_history

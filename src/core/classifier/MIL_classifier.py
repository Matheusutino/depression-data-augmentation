import optuna
from optuna.samplers import TPESampler
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer


class MILDataClassifier:
    def __init__(self, embed_name, random_seed=42, device="cpu"):
        """Inicializa a classe MIL com o nome do modelo de embedding e uma seed"""
        self.embed_name = embed_name
        self.model = SentenceTransformer(embed_name, trust_remote_code=True)
        self.scaler = StandardScaler()
        self.random_seed = random_seed
        self.device = device
    
    def generate_embeddings(self, posts, batch_size = 8):
        """Gera embeddings para os posts de um usuário usando o modelo especificado"""
        return self.model.encode(posts, batch_size=batch_size, show_progress_bar=True)

    def apply_pooling(self, embeddings, pooling_type):
        """Aplica pooling sobre as embeddings geradas"""
        if pooling_type == 'max':
            return np.max(embeddings, axis=0)
        elif pooling_type == 'min':
            return np.min(embeddings, axis=0)
        elif pooling_type == 'mean':
            return np.mean(embeddings, axis=0)
        elif pooling_type == 'sum':
            return np.sum(embeddings, axis=0)
        else:
            raise ValueError("Pooling type not supported")

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

    def train_and_evaluate(self, x_train, x_val, x_test, y_train, y_val, y_test, n_trials=50, pooling_type='average'):
        """Treina o modelo XGBoost utilizando Optuna para otimização dos hiperparâmetros"""
        
        # Gerar embeddings para todos os conjuntos de dados
        x_train_embeds = [self.generate_embeddings(posts) for posts in x_train]
        x_val_embeds = [self.generate_embeddings(posts) for posts in x_val]
        x_test_embeds = [self.generate_embeddings(posts) for posts in x_test]
        
        # Aplicar pooling
        x_train_pool = np.array([self.apply_pooling(embeds, pooling_type) for embeds in x_train_embeds])
        x_val_pool = np.array([self.apply_pooling(embeds, pooling_type) for embeds in x_val_embeds])
        x_test_pool = np.array([self.apply_pooling(embeds, pooling_type) for embeds in x_test_embeds])
        
        # Normalizar embeddings
        x_train_pool = self.scaler.fit_transform(x_train_pool)
        x_val_pool = self.scaler.transform(x_val_pool)
        x_test_pool = self.scaler.transform(x_test_pool)
        
        # Otimizar os hiperparâmetros usando Optuna
        best_params, trials_history = self.optimize_hyperparameters(x_train_pool, y_train, x_val_pool, y_val, n_trials)
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'predictor': 'gpu_predictor',
            'device': self.device ,
            'seed': self.random_seed
        })
        
        # Treinar o modelo final com os melhores parâmetros
        dtrain = xgb.DMatrix(x_train_pool, label=y_train)
        dval = xgb.DMatrix(x_val_pool, label=y_val)
        dtest = xgb.DMatrix(x_test_pool, label=y_test)
        
        final_clf = xgb.train(best_params, dtrain, num_boost_round=1000, 
                        evals=[(dval, 'eval')],
                        early_stopping_rounds=50)
        
        # Avaliar no conjunto de teste
        y_test_pred = final_clf.predict(dtest)
        y_test_pred_binary = (y_test_pred > 0.5).astype(int)
        
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred_binary),
            "precision": precision_score(y_test, y_test_pred_binary),
            "recall": recall_score(y_test, y_test_pred_binary),
            "f1_score": f1_score(y_test, y_test_pred_binary)
        }
        
        result = {
            "best_params": best_params,
            "test_metrics": test_metrics
        }
        
        return result, trials_history

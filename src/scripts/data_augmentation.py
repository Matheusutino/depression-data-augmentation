import re
import os
import pandas as pd
import numpy as np
from src.core.data_augmentation.data_augmentation_pipeline import DataAugmentationPipeline
from src.core.preprocessing import DataPreprocessor
from src.core.utils import read_yaml, save_json, merge_dicts

# Configurações e carregamento de dados
dataset_path = "datasets/instagram"
augmentor_type = "copy"
n = 2
llm_provider = "openai"
model = "gpt-4o-mini"
temperature = 0.5
max_output_tokens = 500
prompt_name = "bdi"
prompt_file = "configs/prompts.yaml"
questionnaire_path = "datasets/questionnaire/questionnaire.csv"

dataset_name = os.path.basename(dataset_path)

# Instanciando a classe DataPreprocessor
preprocessor = DataPreprocessor(dataset_path, questionnaire_path, dataset_name)

# Carregando e processando os dados
datasets = preprocessor.load_and_preprocess_dataset()
questionnaire_filtered = preprocessor.filter_questionnaire_data()

# Agrupando por usuário
grouped_datasets = {key: df.groupby("username").agg(list) for key, df in datasets.items()}

# Processando os dados
processed_data = {
    key: preprocessor.preprocess_data(grouped_datasets[key], questionnaire_filtered)
    for key in ["train", "val", "test"]
}

# Separando os dados de treinamento
train_data = processed_data["train"]
X_train = train_data["X"]
y_train = train_data["y"]
bdi_values_train = train_data["bdi_values"]
bdi_forms_train = train_data["bdi_forms"]
usernames = train_data["usernames"]

# print(bdi_forms_train)

# Carregar prompts
prompt_file_data = read_yaml(prompt_file)
system_prompt = prompt_file_data[prompt_name]["system_prompt"]
user_prompt = prompt_file_data[prompt_name]["user_prompt"]

# Pipeline de aumento de dados
pipeline = DataAugmentationPipeline(
    augmentor_type=augmentor_type,
    llm_provider=llm_provider,
    model=model
)

# Aumentar os dados
train_augmented = pipeline.augment_data(
    X_train,
    y_train,
    BDI_forms=bdi_forms_train,
    BDI_values=bdi_values_train,
    usernames=usernames,
    n=n,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    temperature=temperature,
    max_output_tokens=max_output_tokens,
    min_num_posts=2,
    max_num_posts=4
)

# Combinando os dados originais com os aumentados
train_combined = merge_dicts(train_data, train_augmented)

# Função para salvar os dados
output_path = f"processed_data/{dataset_name}/{augmentor_type}"

# Criar diretório se não existir
os.makedirs(output_path, exist_ok=True)

# Salvar os dicionários processados
save_json(f"{output_path}/train_original.json", train_data)
save_json(f"{output_path}/train_augmented.json", train_augmented)
save_json(f"{output_path}/train_combined.json", train_combined)
save_json(f"{output_path}/val.json", processed_data["val"])
save_json(f"{output_path}/test.json", processed_data["test"])

# # Exibir estatísticas
# total_samples = len(X_train)
# total_augmented = len(new_X)
# label_distribution = dict(zip(*np.unique(new_y, return_counts=True)))

# print(f"Total samples: {total_samples}")
# print(f"Total augmented samples: {total_augmented}")
# print("Label distribution:", label_distribution)


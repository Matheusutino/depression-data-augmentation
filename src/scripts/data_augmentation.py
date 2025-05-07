import argparse
import os
from src.core.data_augmentation.data_augmentation_pipeline import DataAugmentationPipeline
from src.core.preprocessing import DataPreprocessor
from src.core.utils import read_yaml, save_json, merge_dicts

def data_augmentation(dataset_path, 
                      augmentor_type, 
                      n, 
                      llm_provider, 
                      model, 
                      model_type,
                      temperature, 
                      max_output_tokens, 
                      prompt_name, 
                      prompt_file, 
                      questionnaire_path):
    # Configurações e carregamento de dados
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

    # Carregar prompts
    prompt_file_data = read_yaml(prompt_file)
    system_prompt = prompt_file_data[prompt_name]["system_prompt"]
    user_prompt = prompt_file_data[prompt_name]["user_prompt"]

    # Pipeline de aumento de dados
    pipeline = DataAugmentationPipeline(
        augmentor_type=augmentor_type,
        llm_provider=llm_provider,
        model=model,
        model_type=model_type
    )

    # Aumentar os dados
    train_augmented_all = {
        "X": [],
        "y": [],
        "bdi_values": [],
        "bdi_forms": [],
        "usernames": []
    }

    for i in range(1, n + 1):
        train_augmented = pipeline.augment_data(
            X_train,
            y_train,
            BDI_forms=bdi_forms_train,
            BDI_values=bdi_values_train,
            usernames=usernames,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            social_media = dataset_name
        )

        for key, value in train_augmented.items():
            # Para adicionar as novas instâncias, você pode atualizar diretamente o dicionário
            train_augmented_all[key].extend(value)

        # Combinando os dados originais com os aumentados
        train_combined = merge_dicts(train_data, train_augmented_all)
        
        # Função para salvar os dados
        if augmentor_type == "llm":
            model = model.split("/", 1)[1] if "/" in model else model
            output_path = f"processed_data/{dataset_name}/{augmentor_type}/{model}/{prompt_name}/{i}"
        elif augmentor_type == "contextual":
            model = model.split("/", 1)[1] if "/" in model else model
            output_path = f"processed_data/{dataset_name}/{augmentor_type}/{model}/{i}"
        else:
            output_path = f"processed_data/{dataset_name}/{augmentor_type}/{i}"

        # Criar diretório se não existir
        os.makedirs(output_path, exist_ok=True)

        # Salvar os dicionários processados
        save_json(f"{output_path}/train_original.json", train_data)
        save_json(f"{output_path}/train_augmented.json", train_augmented_all)
        save_json(f"{output_path}/train_combined.json", train_combined)
        save_json(f"{output_path}/val_original.json", processed_data["val"])
        save_json(f"{output_path}/test_original.json", processed_data["test"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Augmentation Pipeline")

    # Definindo os parâmetros a serem passados pela linha de comando
    parser.add_argument("dataset_path", type=str, help="Caminho para o dataset")
    parser.add_argument("augmentor_type", type=str, choices=["copy", "contextual", "llm"], help="Tipo de augmentor a ser utilizado")
    parser.add_argument("--n", type=int, default=5, help="Número de aumentos de dados")
    parser.add_argument("--llm_provider", type=str, help="Fornecedor de LLM (ex: openai)")
    parser.add_argument("--model", type=str, help="Modelo LLM (ex: gpt-4o-mini)")
    parser.add_argument("--model_type", type=str, help="Modelo para o augmentor contextual (ex: 'BERT')")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperatura para geração do modelo")
    parser.add_argument("--max_output_tokens", type=int, default= 16384, help="Máximo de tokens de saída")
    parser.add_argument("--prompt_name", type=str, default = "no_bdi", help="Nome do prompt")
    parser.add_argument("--prompt_file", type=str, default="configs/prompts.yaml", help="Caminho para o arquivo de prompts YAML")
    parser.add_argument("--questionnaire_path", type=str, default="datasets/questionnaire/questionnaire.csv", help="Caminho para o arquivo do questionário")

    # Parse dos argumentos
    args = parser.parse_args()

    # Chama a função de aumento de dados com os parâmetros fornecidos
    data_augmentation(
        args.dataset_path,
        args.augmentor_type,
        args.n,
        args.llm_provider,
        args.model,
        args.model_type,
        args.temperature,
        args.max_output_tokens,
        args.prompt_name,
        args.prompt_file,
        args.questionnaire_path
    )

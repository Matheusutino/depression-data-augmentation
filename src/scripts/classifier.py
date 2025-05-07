from src.core.classifier.classifier import classifier
from src.core.utils import get_subdirectories

dataset_processed_paths = ["processed_data/instagram/llm/gemini-2.0-flash/bdi",
                           "processed_data/instagram/llm/gemini-2.0-flash/no_bdi",
                        #    "processed_data/instagram/llm/DeepSeek-R1/no_bdi",
                           "processed_data/instagram/llm/Llama-4-Maverick-17B-128E-Instruct-FP8/no_bdi",
                           "processed_data/instagram/llm/Llama-4-Maverick-17B-128E-Instruct-FP8/bdi",
                           "processed_data/instagram/llm/llama-3.3-70b-instruct/bdi",
                           "processed_data/instagram/llm/llama-3.3-70b-instruct/no_bdi",
                           "processed_data/instagram/llm/Mistral-Small-3.1-24B-Instruct-2503/bdi",
                           "processed_data/instagram/llm/Mistral-Small-3.1-24B-Instruct-2503/no_bdi",
                            "processed_data/instagram/copy", 
                            "processed_data/instagram/llm/dolphin3:8b/no_bdi", 
                            "processed_data/instagram/contextual/bert-base-multilingual-cased",
                            "processed_data/instagram/contextual/bert-large-portuguese-cased",
                            "processed_data/instagram/llm/dolphin3:8b/bdi"]
types_classifier = ["SIL", "MIL"]
embed_names = ["ibm-granite/granite-embedding-107m-multilingual", "paraphrase-multilingual-mpnet-base-v2", "intfloat/multilingual-e5-base", "distiluse-base-multilingual-cased"]
poolings_type = ["mean"]
n_trials = 20

for dataset_processed_path in dataset_processed_paths:
    all_subdirectories = get_subdirectories(dataset_processed_path)
    for subdirectory in all_subdirectories:
        for type_classifier in types_classifier:
            for pooling_type in poolings_type:
                for embed_name in embed_names:
                    train_original_dataset_path = f"{subdirectory}/train_original.json"
                    train_augmented_dataset_path = f"{subdirectory}/train_augmented.json"
                    train_combined_dataset_path = f"{subdirectory}/train_combined.json"
                    train_paths = [train_original_dataset_path, train_augmented_dataset_path, train_combined_dataset_path]
                    val_dataset_path = f"{subdirectory}/val_original.json"
                    test_dataset_path = f"{subdirectory}/test_original.json"

                    for train_path in train_paths:
                        try:
                            classifier(
                                train_dataset_path=train_path,
                                val_dataset_path=val_dataset_path,
                                test_dataset_path=test_dataset_path,
                                type_classifier=type_classifier,
                                embed_name=embed_name,
                                pooling_type=pooling_type,
                                n_trials=n_trials
                            )
                        except Exception as e:
                            print(f"Error processing {train_path} with classifier {type_classifier}: {e}")

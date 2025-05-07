import os
import argparse
from src.core.utils import save_json, read_json, get_output_path, check_directory_exists, create_directory
from src.core.classifier.classifier_factory import ClassifierFactory


def classifier(train_dataset_path, 
               val_dataset_path, 
               test_dataset_path, 
               type_classifier, 
               embed_name="distiluse-base-multilingual-cased", 
               pooling_type='mean', 
               n_trials=50):
    # Output
    output_path = get_output_path(train_dataset_path, type_classifier=type_classifier, embed_name=embed_name.replace("/", "_"), pooling_type=pooling_type)
    print(output_path)
    check_directory_exists(output_path)
    
    # Load data
    dataset_train = read_json(train_dataset_path)
    dataset_val = read_json(val_dataset_path)
    dataset_test = read_json(test_dataset_path)
    
    # Instantiate the DataClassifier class
    classifier = ClassifierFactory.create_classifier(type_classifier, embed_name)
    
    # Train and evaluate the model
    result, trials_history = classifier.train_and_evaluate(
        x_train=dataset_train["X"],
        x_val=dataset_val["X"],
        x_test=dataset_test["X"],
        y_train=dataset_train["y"],
        y_val=dataset_val["y"],
        y_test=dataset_test["y"],
        n_trials=n_trials,
        pooling_type=pooling_type
    )
    
    create_directory(output_path)
    save_json(os.path.join(output_path, "results.json"), result)
    save_json(os.path.join(output_path, "trials_history.json"), trials_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a data classification model.")
    parser.add_argument("train_dataset_path", type=str, help="Path to the training dataset.")
    parser.add_argument("val_dataset_path", type=str, help="Path to the validation dataset.")
    parser.add_argument("test_dataset_path", type=str, help="Path to the test dataset.")
    parser.add_argument("type_classifier", type=str, help="SIL or MIL.")
    parser.add_argument("--embed_name", type=str, default="multi-qa-mpnet-base-dot-v1", help="Embedding model name.")
    parser.add_argument("--pooling_type", type=str, default="mean", help="Pooling type to be used.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials for hyperparameter optimization.")
    
    args = parser.parse_args()

    
    classifier(
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        test_dataset_path=args.test_dataset_path,
        type_classifier=args.type_classifier,
        embed_name=args.embed_name,
        pooling_type=args.pooling_type,
        n_trials=args.n_trials
    )
from src.core.classifier.SIL_classifier import SILDataClassifier
from src.core.classifier.MIL_classifier import MILDataClassifier

class ClassifierFactory:
    """Fábrica para instanciar um classificador baseado em uma string."""
    
    classifiers = {
        "SIL": SILDataClassifier,
        "MIL": MILDataClassifier
    }

    @staticmethod
    def create_classifier(type_classifier, embed_name):
        if type_classifier in ClassifierFactory.classifiers:
            return ClassifierFactory.classifiers[type_classifier](embed_name)
        else:
            raise ValueError(f"Tipo de classificador desconhecido: {type_classifier}")

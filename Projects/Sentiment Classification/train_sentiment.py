from Modules.Models import Classifier
from Modules.Config import Config

def setup(classifier_config: Config) -> None:
    classifier = Classifier(classifier_config)
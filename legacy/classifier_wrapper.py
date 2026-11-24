from pandas import DataFrame
from tqdm import tqdm

from config import Config

from llm_classifier import LLMClassifier
from plm_classifier import PLMClassifier

class ClassifierWrapper:

    # METTRE LA BONNE VALEUR ci-dessous en fonction de la méthode utilisée
    METHOD: str = 'PLMFT'  # or 'PLMFT' (for Pretrained Language Model Fine-Tuning)

    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if self.METHOD == 'LLM':
            self.classifier = LLMClassifier(cfg)
        elif self.METHOD == 'PLMFT':
            self.classifier = PLMClassifier(cfg)
        else:
            raise ValueError(f"METHOD doit être 'LLM' ou 'PLMFT', reçu: {self.METHOD}")


    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        """
        :param train_data:
        :param val_data:
        :param device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu
        :return:
        """
        # Entraîner le modèle si PLMFT, sinon LLM en zéro-shot ne nécessite pas d'entraînement
        if self.METHOD == 'PLMFT':
            self.classifier.train(train_data, val_data, device)
        # Pour LLM, pas d'entraînement nécessaire (zéro-shot)



    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def predict(self, texts: list[str], device: int) -> list[dict]:
        """
        :param texts:
        :param device: device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu à utiliser
        :return:
        """
        if self.METHOD == 'LLM':
            # LLM: traiter texte par texte avec la barre de progression
            all_opinions = []
            for text in tqdm(texts, desc="Predicting (LLM)"):
                opinions = self.classifier.predict(text)
                all_opinions.append(opinions)
            return all_opinions
        elif self.METHOD == 'PLMFT':
            # PLMFT: traiter par batch
            return self.classifier.predict(texts, device)
        else:
            raise ValueError(f"METHOD doit être 'LLM' ou 'PLMFT', reçu: {self.METHOD}")

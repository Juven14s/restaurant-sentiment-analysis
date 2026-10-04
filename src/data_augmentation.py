"""
Module de data augmentation pour les avis de restaurants.
Crée des paraphrases synthétiques en utilisant Ollama/Qwen.
"""

from openai import OpenAI
from tqdm import tqdm
from config import Config
from typing import List, Dict
import random

PARAPHRASE_PROMPT = """Tu es un expert en rédaction française. Ta tâche est de paraphraser cet avis de restaurant de manière naturelle, en gardant exactement le même sens et les mêmes opinions.

Avis original:
"{text}"

Génère UNE SEULE paraphrase naturelle de cet avis (sans explication):
"""


class DataAugmenter:
    """Augmente les données d'entraînement en créant des paraphrases"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm_client = OpenAI(
            base_url=cfg.ollama_url,
            api_key="ollama"
        )
        self.model_name = 'qwen3:1.7b'
        self.model_options = {
            'max_tokens': 300,
            'temperature': 0.7,
            'top_p': 0.9,
        }

    def paraphrase_text(self, text: str) -> str:
        """
        Génère une paraphrase d'un texte via Ollama

        Args:
            text: Texte à paraphraser

        Returns:
            Paraphrase du texte
        """
        try:
            prompt = PARAPHRASE_PROMPT.format(text=text)
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Tu es un expert en paraphrase de texte français. Sois naturel et concis."},
                    {"role": "user", "content": prompt}
                ],
                **self.model_options
            )

            paraphrase = response.choices[0].message.content.strip()
            return paraphrase
        except Exception as e:
            print(f"Erreur lors de la paraphrase: {e}")
            return text

    def augment_data(self, train_data: List[Dict], num_paraphrases: int = 1, augmentation_ratio: float = 0.5) -> List[Dict]:
        """
        Augmente les données d'entraînement en créant des paraphrases sur un sous-ensemble aléatoire

        Args:
            train_data: Données d'entraînement originales
            num_paraphrases: Nombre de paraphrases par avis sélectionné (1 ou 2)
            augmentation_ratio: Ratio des avis à paraphraser (0.0 à 1.0)
                               - 0.5 = 50% des avis seront paraphrasés
                               - 1.0 = 100% des avis seront paraphrasés

        Returns:
            Données originales + données augmentées
        """
        print(f"\n📝 Data Augmentation:")
        print(f"   Avis originaux: {len(train_data)}")
        print(f"   Ratio augmentation: {augmentation_ratio*100:.0f}%")
        print(f"   Paraphrases par avis: {num_paraphrases}")

        # Sélectionner aléatoirement un sous-ensemble d'avis à augmenter
        num_to_augment = max(1, int(len(train_data) * augmentation_ratio))
        indices_to_augment = random.sample(range(len(train_data)), num_to_augment)

        augmented_data = list(train_data)  # Copie les données originales

        for idx in tqdm(indices_to_augment, desc="Paraphrasing"):
            item = train_data[idx]
            original_text = item['Avis']

            # Créer N paraphrases
            for _ in range(num_paraphrases):
                paraphrase = self.paraphrase_text(original_text)

                # Créer un nouvel item avec la paraphrase et les mêmes labels
                augmented_item = {
                    'Avis': paraphrase,
                    'Prix': item['Prix'],
                    'Cuisine': item['Cuisine'],
                    'Service': item['Service']
                }
                augmented_data.append(augmented_item)

        print(f"✓ Total après augmentation: {len(augmented_data)} avis")
        print(f"  (+{len(augmented_data) - len(train_data)} avis synthétiques)\n")

        return augmented_data


def augment_if_needed(train_data: List[Dict], cfg: Config, use_augmentation: bool = True, augmentation_ratio: float = 0.5, num_paraphrases: int = 1) -> List[Dict]:
    """
    Fonction utilitaire pour augmenter les données optionnellement

    Args:
        train_data: Données d'entraînement
        cfg: Configuration
        use_augmentation: Activer la data augmentation
        augmentation_ratio: Ratio des avis à paraphraser (0.0 à 1.0)
        num_paraphrases: Nombre de paraphrases par avis (1 ou 2)

    Returns:
        Données augmentées ou originales
    """
    if not use_augmentation:
        return train_data

    augmenter = DataAugmenter(cfg)
    return augmenter.augment_data(train_data, num_paraphrases=num_paraphrases, augmentation_ratio=augmentation_ratio)

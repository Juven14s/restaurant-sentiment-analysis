import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from config import Config
from data_augmentation import augment_if_needed


class OpinionDataset(Dataset):
    """Dataset pour charger les données d'avis avec les labels d'opinions"""

    def __init__(self, data: list[dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aspects = ['Prix', 'Cuisine', 'Service']
        self.opinion_to_label = {
            'Positive': 0,
            'Négative': 1,
            'Neutre': 2,
            'NE': 3
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['Avis']

        # Tokenize le texte
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Créer les labels pour chaque aspect
        labels = {}
        for aspect in self.aspects:
            opinion = item.get(aspect, 'NE')
            labels[aspect] = self.opinion_to_label.get(opinion, 3)  # Default to NE (3)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }


class PLMClassifier:
    """Classifieur basé sur un modèle pré-entraîné masqué (CamemBERT)"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device('cuda' if cfg.device >= 0 else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')
        self.base_model = AutoModel.from_pretrained('almanach/camembert-base')

        # Ajouter des heads de classification pour chaque aspect
        self.aspects = ['Prix', 'Cuisine', 'Service']
        self.num_classes = 4  # Positive, Négative, Neutre, NE
        self.dropout = nn.Dropout(0.3)

        # Créer un classificateur simple pour chaque aspect
        self.classifiers = nn.ModuleDict({
            aspect: nn.Linear(self.base_model.config.hidden_size, self.num_classes)
            for aspect in self.aspects
        })

        self.model = nn.Module()
        self.model.base_model = self.base_model
        self.model.dropout = self.dropout
        self.model.classifiers = self.classifiers

        self.model.to(self.device)

        # Pour la prédiction
        self.label_to_opinion = {
            0: 'Positive',
            1: 'Négative',
            2: 'Neutre',
            3: 'NE'
        }

    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        """
        Fine-tune le modèle sur les données d'entraînement

        :param train_data: données d'entraînement
        :param val_data: données de validation
        :param device: GPU/CPU device
        """
        print("Préparation du fine-tuning...")

        # Data augmentation -> Augmente les données d'entraînement
        train_data = augment_if_needed(train_data, self.cfg, use_augmentation=False)

        # Créer les datasets et dataloaders
        train_dataset = OpinionDataset(train_data, self.tokenizer)
        val_dataset = OpinionDataset(val_data, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)

        # Optimizer et scheduler avec learning rate optimal
        optimizer = AdamW(self.model.parameters(), lr=3e-5, weight_decay=0.01)
        num_epochs = 3
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # CrossEntropyLoss avec label smoothing pour meilleure généralisation
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        best_val_loss = float('inf')

        # Boucle d'entraînement
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                optimizer.zero_grad()

                # Forward pass
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                pooled_output = self.dropout(pooled_output)

                # Calculer la loss pour chaque aspect
                loss = 0
                for aspect in self.aspects:
                    aspect_labels = torch.tensor(
                        [labels[aspect][i].item() if isinstance(labels[aspect][i], torch.Tensor)
                         else labels[aspect][i] for i in range(len(labels[aspect]))],
                        dtype=torch.long
                    ).to(self.device)
                    logits = self.classifiers[aspect](pooled_output)
                    aspect_loss = criterion(logits, aspect_labels)
                    loss += aspect_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Training loss: {avg_train_loss:.4f}")

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels']

                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    pooled_output = outputs.last_hidden_state[:, 0, :]
                    pooled_output = self.dropout(pooled_output)

                    loss = 0
                    for aspect in self.aspects:
                        aspect_labels = torch.tensor(
                            [labels[aspect][i].item() if isinstance(labels[aspect][i], torch.Tensor)
                             else labels[aspect][i] for i in range(len(labels[aspect]))],
                            dtype=torch.long
                        ).to(self.device)
                        logits = self.classifiers[aspect](pooled_output)
                        aspect_loss = criterion(logits, aspect_labels)
                        loss += aspect_loss

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"✓ Nouvelle meilleure validation loss: {best_val_loss:.4f}")

    def predict(self, texts: list[str], device: int) -> list[dict]:
        """
        Prédire les opinions pour une liste de textes

        :param texts: liste des textes d'avis
        :param device: GPU/CPU device
        :return: liste de dictionnaires avec les opinions prédites
        """
        self.model.eval()
        predictions = []

        dataset = OpinionDataset([{'Avis': text} for text in texts], self.tokenizer)
        loader = DataLoader(dataset, batch_size=16)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                pooled_output = outputs.last_hidden_state[:, 0, :]
                pooled_output = self.dropout(pooled_output)

                batch_preds = []
                for i in range(len(input_ids)):
                    pred_dict = {}
                    for aspect in self.aspects:
                        logits = self.classifiers[aspect](pooled_output[i:i+1])
                        pred_label = torch.argmax(logits, dim=1).item()
                        pred_dict[aspect] = self.label_to_opinion[pred_label]
                    batch_preds.append(pred_dict)

                predictions.extend(batch_preds)

        return predictions

    def _compute_class_weights(self, train_data: list[dict]) -> torch.Tensor:
        """
        Calcule les poids des classes pour gérer le déséquilibre
        """
        aspects = ['Prix', 'Cuisine', 'Service']
        opinion_to_idx = {'Positive': 0, 'Négative': 1, 'Neutre': 2, 'NE': 3}

        # Compter les occurrences de chaque classe
        class_counts = torch.zeros(4)
        for item in train_data:
            for aspect in aspects:
                opinion = item.get(aspect, 'NE')
                idx = opinion_to_idx.get(opinion, 3)
                class_counts[idx] += 1

        # Calculer les poids (inverse frequency)
        total = class_counts.sum()
        weights = total / (4 * class_counts + 1e-6)  # +1e-6 pour éviter division par 0
        weights = weights / weights.sum() * 4  # Normaliser

        print(f"Poids des classes: {weights.tolist()}")
        return weights

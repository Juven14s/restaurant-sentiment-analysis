from jinja2 import Template
from openai import OpenAI
import re
import json


from config import Config


_PROMPT_TEMPLATE = """Vous êtes un expert en analyse d'opinions. Analysez l'avis suivant et classifiez l'opinion globale de l'auteur pour chacun des 3 aspects prédéfinis.

AVIS:
"{{text}}"

DIMENSIONS À ANALYSER:
1. Prix: opinions concernant les prix des plats et des boissons
2. Cuisine: opinions concernant la qualité et la quantité de la nourriture et des boissons proposées
3. Service: opinions concernant la qualité et l'efficacité du service et de l'accueil

RÈGLES DE CLASSIFICATION:
- Positive: l'avis contient une ou plusieurs opinions positives sur l'aspect ET AUCUNE opinion négative
- Négative: l'avis contient une ou plusieurs opinions négatives sur l'aspect ET AUCUNE opinion positive
- Neutre: l'avis exprime à la fois au moins une opinion positive ET une opinion négative sur l'aspect
- NE (Non Exprimée): l'avis ne contient aucune opinion sur l'aspect

RÉSULTAT:
Retournez UNIQUEMENT un objet JSON valide au format:
{"Prix": "...", "Cuisine": "...", "Service": "..."}

Les valeurs doivent être exactement: Positive, Négative, Neutre, ou NE"""



class LLMClassifier:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        base_url = cfg.ollama_url
        self.llmclient = OpenAI(
            base_url=base_url,
            api_key="ollama"
        )
        self.model_name = 'gemma3:1b'
        self.model_options = {
            'max_tokens': 500,
            'temperature': 0.1,
            'top_p': 0.9,
        }
        self.jtemplate = Template(_PROMPT_TEMPLATE)


    def predict(self, text: str) -> dict[str,str]:
        """
        Lance au LLM une requête contenant le texte de l'avis et les instructions pour extraire
        les opinions sur les aspects sous forme d'objet json
        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 4 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        prompt = self.jtemplate.render(text=text)
        response = self.llmclient.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.model_options['temperature'],
            top_p=self.model_options['top_p'],
            max_tokens=self.model_options['max_tokens']
        )
        response_text = response.choices[0].message.content
        jresp = self.parse_json_response(response_text)
        return jresp

    def parse_json_response(self, response: str) -> dict[str, str] | None:
        m = re.findall(r"\{[^\{\}]+\}", response, re.DOTALL)
        if m:
            try:
                jresp = json.loads(m[0])
                for aspect, opinion in jresp.items():
                    if "non exprim" in opinion.lower():
                        jresp[aspect] = "NE"
                return jresp
            except:
                return None
        else:
            return None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
from transformers import MarianMTModel, MarianTokenizer
from google import genai

# Local translator to convert french text to english text before feeding it to the intent detection.
class LocalTranslator:
    def __init__(self):
        """Initializes a lightweight, local French-to-English translation model."""
        print("[LocalTranslator] Loading offline translation model...")
        model_name = "Helsinki-NLP/opus-mt-fr-en"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        print("[LocalTranslator] Model loaded successfully.")

    def translate_to_english(self, text: str) -> str:
        """Translates French text to English locally without any API calls."""
        try:
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)

            # Generate the translation
            translated_tokens = self.model.generate(**inputs)

            # Decode back to a string
            translation = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translation.strip()
        except Exception as e:
            print(f"[Translation Error]: {e}")
            return text # Fallback to original text if it fails
    

class NLGEngine:
    def __init__(self, api_keys: list[str], models: list[str] = None):
        """
        Initializes the NLG Engine with multiple fallback keys and models.
        """
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.models = models if models else [
            'gemini-2.5-flash',
            'gemini-2.0-flash',
            'gemini-1.5-pro'
        ]

        # ── PRE-CANNED TEMPLATE RESPONSES ──
        # Placeholders match the lowercase keys usually returned in your api_json.
        self.templates = {
            "English": {
                "OOS": "I'm sorry, I don't know how to help with that.",
                "Greeting": "Hello! How can I help you today?",
                "Goodbye": "Goodbye! Have a great day.",
                "SetTimer": "I've set a timer for {duration}.",
                "GetWeather": "Currently in {location}, it is {temperature} degrees.",
                "GetTopHeadlines": "Here are the top news headlines for today.",
                "GetTopicNews": "Here is the latest news regarding {topic}.",
                "GetPublisherHeadlines": "Here are the latest headlines from {source}.",
                "GetRegionNews": "Here is the latest news for {region}.",
                "GetGameScore": "I fetched the latest game score for the {team}.",
                "GetTeamStanding": "Here are the current league standings for the {team}.",
                "GetLeagueSchedule": "Here is the upcoming schedule.",
                "GetTeamStats": "Here are the latest statistics for the {team}.",
                "SetOvenTemperature": "I have set the oven to {temperature}.",
                "ToggleLights": "I have updated the kitchen lights.",
                "EditShoppingList": "I have updated your shopping list with {food_item}.",
                "QueryNutrition": "Here is the nutritional information for {food_item}.",
                "SetCaloricGoal": "Your caloric goal has been set to {calorie_value}.",
                "NotifyOvenReady": "The oven is preheated and ready."
            },
            "French": {
                "OOS": "Désolé, je ne sais pas comment vous aider avec ça.",
                "Greeting": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
                "Goodbye": "Au revoir ! Passez une bonne journée.",
                "SetTimer": "J'ai réglé un minuteur pour {duration}.",
                "GetWeather": "Actuellement à {location}, il fait {temperature} degrés.",
                "GetTopHeadlines": "Voici les gros titres de l'actualité d'aujourd'hui.",
                "GetTopicNews": "Voici les dernières nouvelles concernant {topic}.",
                "GetPublisherHeadlines": "Voici les derniers gros titres de {source}.",
                "GetRegionNews": "Voici les dernières nouvelles pour la région de {region}.",
                "GetGameScore": "J'ai récupéré le dernier score du match pour l'équipe {team}.",
                "GetTeamStanding": "Voici le classement actuel pour l'équipe {team}.",
                "GetLeagueSchedule": "Voici le calendrier à venir.",
                "GetTeamStats": "Voici les dernières statistiques pour l'équipe {team}.",
                "SetOvenTemperature": "J'ai réglé le four à {temperature}.",
                "ToggleLights": "J'ai mis à jour les lumières de la cuisine.",
                "EditShoppingList": "J'ai mis à jour votre liste de courses avec {food_item}.",
                "QueryNutrition": "Voici les informations nutritionnelles pour {food_item}.",
                "SetCaloricGoal": "Votre objectif calorique a été fixé à {calorie_value}.",
                "NotifyOvenReady": "Le four est préchauffé et prêt."
            }
        }

    def get_client(self):
        """Creates a client using the currently active API key."""
        return genai.Client(api_key=self.api_keys[self.current_key_idx])

    def rotate_key(self):
        """Moves to the next API key in the list."""
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        print(f"[NLG System] Switched to API key index {self.current_key_idx}")

    def _get_template_fallback(self, api_json: dict, language: str) -> str:
        """
        Retrieves the pre-canned template and safely injects JSON data.
        """
        lang_dict = self.templates.get(language, self.templates["English"])

        # 1. Check if the backend threw an error first
        if "error" in api_json:
            err_msg = api_json["error"]
            if language == "French":
                return f"Désolé, j'ai rencontré un problème : {err_msg}"
            return f"I'm sorry, I ran into an issue: {err_msg}"

        # 2. Extract intent and grab the template string
        intent = api_json.get("intent", "OOS")
        template_str = lang_dict.get(intent, lang_dict["OOS"])

        # 3. Safely format the template (missing keys will just show up as 'unknown')
        # We flatten lists so ['Lakers'] becomes 'Lakers' in the sentence.
        clean_json = {k.lower(): (v[0] if isinstance(v, list) and v else v)
                      for k, v in api_json.items()}
        safe_format_dict = collections.defaultdict(lambda: "unknown", clean_json)

        return template_str.format_map(safe_format_dict)

    def generate_natural_response(self, user_message: str, api_json: dict, language: str = "English") -> str:
        """
        Generates a response, cycling through keys and models if rate limits occur.
        Falls back to templates if all LLM attempts fail.
        """
        prompt = f"""
        You are Atlas, a smart, helpful morning voice assistant.

        The user just said: "{user_message}"

        Your backend system processed this request and returned the following data:
        {api_json}

        Rules for your response:
        1. Formulate a natural, conversational response based ONLY on the data provided.
        2. If the data contains an "error", apologize gracefully and explain the issue to the user.
        3. Keep it concise and conversational (1-3 sentences max) because this will be spoken out loud.
        4. NEVER mention "JSON", "backend", "API", or "the data". Speak as if you just know the answer.
        5. You MUST reply entirely in the following language: {language}.
        6. If the user queries about nutrition info, keep in mind that the results in the JSON response are for
        100g by default. Do any necessary conversions to return the correct calorie count for the queried quantity.
        """

        # Outer Loop: Try each model in our fallback chain
        for model_name in self.models:

            # Inner Loop: Try every API key we have for this model
            for _ in range(len(self.api_keys)):
                client = self.get_client()

                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    return response.text.strip()

                except Exception as e:
                    error_str = str(e)
                    print(f"[NLG Warning] Key Index {self.current_key_idx} | Model: {model_name} | Error: {error_str}")

                    if "429" in error_str or "quota" in error_str.lower() or "exhausted" in error_str.lower():
                        print("[NLG System] Rate limit hit. Rotating to next API key...")
                        self.rotate_key()
                        continue

                    elif "404" in error_str:
                        print(f"[NLG System] Model {model_name} is unavailable. Falling back to next model...")
                        break

                    else:
                        print("[NLG System] Unhandled error. Rotating key just in case...")
                        self.rotate_key()

        # ── IF WE REACH THIS POINT, THE LLM HAS FAILED ENTIRELY ──
        print("[NLG Fatal] All LLM methods failed. Generating pre-canned template response.")
        return self._get_template_fallback(api_json, language)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Fulfilment.BasicModule import Greeting, Goodbye, OOS, SetTimer
from Fulfilment.WeatherModule import GetWeather
from Fulfilment.KitchenModule import ToggleLights, SetOvenTemperature, EditShoppingList, SetCaloricGoal, NotifyOvenReady, get_nutrition
from Fulfilment.ESPNModule import GetGameScore, GetTeamStanding, GetLeagueSchedule
from Fulfilment.NewsModule import GetTopHeadlines, GetTopicNews, GetPublisherHeadlines, GetRegionNews

class CommandProcessor:
    def __init__(self, kitchen):
        self.kitchen = kitchen

    def process(self, prediction: dict, language: str = "English") -> dict:
        intent = prediction.get("intent")
        raw_slots = prediction.get("slots", {})

        # Wrap strings in a list safely without splitting words apart
        slots = {k: [v] if isinstance(v, str) else v for k, v in raw_slots.items()}
        slots["_LANGUAGE"] = language

        try:
            # ── Conversational Intents ──
            if intent == "Greeting": return Greeting(slots)
            elif intent == "Goodbye": return Goodbye(slots)
            elif intent == "OOS": return OOS(slots)

            # ── Weather ──
            elif intent == "GetWeather": return GetWeather(slots)

            #Timer
            elif intent == "SetTimer": return SetTimer(slots, self.kitchen)

            # ── Sports Domain ──
            elif intent == "GetGameScore": return GetGameScore(slots)
            elif intent == "GetTeamStanding": return GetTeamStanding(slots)
            elif intent == "GetLeagueSchedule": return GetLeagueSchedule(slots)

            # ── News Domain ──
            elif intent == "GetTopHeadlines": return GetTopHeadlines(slots)
            elif intent == "GetTopicNews": return GetTopicNews(slots)
            elif intent == "GetPublisherHeadlines": return GetPublisherHeadlines(slots)
            elif intent == "GetRegionNews": return GetRegionNews(slots)

            elif intent == "ToggleLights": return ToggleLights(slots, self.kitchen)
            elif intent == "SetOvenTemperature": return SetOvenTemperature(slots, self.kitchen)
            elif intent == "EditShoppingList": return EditShoppingList(slots, self.kitchen)
            elif intent == "SetCaloricGoal": return SetCaloricGoal(slots, self.kitchen)

            elif intent == "NotifyOvenReady": return NotifyOvenReady(slots, self.kitchen)
            elif intent == "QueryNutrition":
                # 1. Safely extract the food item string from the slots list
                food_item = slots.get("FOOD_ITEM", [""])[0]

                # 2. Handle missing slots
                if not food_item:
                    return {"intent": "QueryNutrition", "error": "No food item was recognized in the command."}

                # 3. Call your specific function directly
                result = get_nutrition(food_item)

                # 4. Attach the intent label
                result["intent"] = "QueryNutrition"

                return result

            # ── Unhandled Intents ──
            else:
                return {"intent": intent, "error": f"API handler missing for {intent}."}

        except Exception as e:
            return {"intent": intent, "error": f"Error processing {intent}: {str(e)}"}
        
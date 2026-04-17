#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import threading
import time
import sys
import re
from Fulfilment.Helpers import _unwrap

class Kitchen:
    """
    Thread-safe kitchen controller for oven, lights, and shopping list.
    """
    def __init__(self, tts_engine = None):
        self._lock           = threading.Lock()
        self.shopping_list   = []
        self.item_calories   = {}
        self.current_calories= 0


        self.oven_temp       = 0    # Current temperature
        self.target_temp     = 0    # Target temperature
        self.oven_on         = False
        self.lights_on       = False
        self.timer_remaining = 0
        self.timer_label     = ""
        self.caloric_goal    = 0

        # NEW: Flags for the notification system
        self.notify_user_when_ready = True
        self.ready_audio_path       = None
        self._stop_heating          = threading.Event()

        self.tts_engine = tts_engine

    def set_lights(self, state: bool):
        with self._lock:
            self.lights_on = bool(state)

    def set_oven(self, state: bool, target_temp: int):
        with self._lock:
            current_on   = self.oven_on
            current_temp = self.oven_temp

            # 1. Handle Turning OFF
            if not state:
                if not current_on:
                    return "Oven is already off."
                self.oven_on = False
                self.target_temp = 0
                self.notify_user_when_ready = False
                threading.Thread(target=notify_oven, args=(self, 0), daemon=True).start()
                return "Oven turning off — cooling down."

            # 2. Handle Turning ON or Adjusting
            self.oven_on = True
            self.target_temp = target_temp

            if current_on and current_temp == target_temp:
                return f"Oven is already on and at {target_temp}°F."

            # Trigger notification only if we are actively heating up
            if target_temp > current_temp:
                self.notify_user_when_ready = True
            else:
                self.notify_user_when_ready = False

            # Spawn the new thread. It will automatically override the old one
            # because the old thread will see the new target_temp and abort.
            threading.Thread(target=notify_oven, args=(self, target_temp), daemon=True).start()

            if current_on:
                return f"Oven adjusting from {current_temp}°F to {target_temp}°F."
            else:
                return f"Oven turning on — heating from {current_temp}°F to {target_temp}°F."

    def add_shopping_item(self, item: str, calories: float = 0.0):
        """Add an item to the shopping list with its calorie count."""
        with self._lock:
            if not item:
                return ", ".join(self.shopping_list)

            # Normalize to lowercase so "Apple" and "apple" match
            clean_item = item.lower().strip()

            if clean_item not in self.shopping_list:
                self.shopping_list.append(clean_item)
                self.item_calories[clean_item] = calories
                self.current_calories += calories
            return ", ".join(self.shopping_list)

    def remove_shopping_item(self, item: str):
        """Remove an item from the shopping list and subtract its calories."""
        with self._lock:
            if not item:
                return ", ".join(self.shopping_list)

            clean_item = item.lower().strip()

            try:
                self.shopping_list.remove(clean_item)
                # Subtract the item's calories from the running total
                cal = self.item_calories.pop(clean_item, 0)
                self.current_calories -= cal
                if self.current_calories < 0:
                    self.current_calories = 0
            except ValueError:
                pass  # Item wasn't in the list

            return ", ".join(self.shopping_list)

def notify_oven(kitchen: Kitchen, target_temp: int):
    """
    Simulates the oven heating or cooling by 10 degrees every 1 second.
    Robust against thread collisions.
    """
    with kitchen._lock:
        current_temp = kitchen.oven_temp

    direction = 1 if current_temp < target_temp else -1
    step = 10 * direction

    while True:
        time.sleep(1) # Sleep first, then check state

        with kitchen._lock:
            # ── 1. ABORT CONDITIONS ──
            # If another thread changed the target, or the oven was turned off, abort this thread.
            if kitchen.target_temp != target_temp:
                print(f"[Oven Thread] Aborting: Target changed from {target_temp} to {kitchen.target_temp}")
                sys.stdout.flush()
                return

            # ── 2. CALCULATE TEMPERATURE ──
            current_temp = kitchen.oven_temp

            if current_temp == target_temp:
                break # Already at target

            next_temp = current_temp + step

            # Clamp to prevent overshooting
            if direction == 1 and next_temp > target_temp:
                next_temp = target_temp
            elif direction == -1 and next_temp < target_temp:
                next_temp = target_temp

            kitchen.oven_temp = next_temp
            current_temp = next_temp

        print(f"[Oven Hardware] Current Temp: {current_temp}°F")
        sys.stdout.flush()

        # Stop loop if target reached
        if current_temp == target_temp:
            break

    print(f"[Oven Hardware] Target of {target_temp}°F reached.")
    sys.stdout.flush()

    # ── TTS Notification Trigger ──
    if target_temp > 0:
        should_notify = False
        with kitchen._lock:
            if kitchen.notify_user_when_ready and kitchen.oven_on and kitchen.target_temp == target_temp:
                should_notify = True
                kitchen.notify_user_when_ready = False # Reset flag so it only triggers once

        if should_notify:
            print("🔊 TRIGGERING TTS: Oven is ready!")
            try:
                if kitchen.tts_engine:
                    audio_file = kitchen.tts_engine.generate_audio("The oven has reached its target temperature and is now ready.")
                    with kitchen._lock:
                        kitchen.ready_audio_path = audio_file
            except Exception as e:
                print(f"[Oven TTS Error]: {e}")

# Convenience factory for external import
def create_kitchen(tts_engine=None):
    return Kitchen(tts_engine=tts_engine)

def get_nutrition(food_item, grams=100):
    """
    Query the USDA FoodData Central API.
    Falls back to a robust local offline database if the API fails or is unreachable.
    """

    # ── 1. ROBUST LOCAL FALLBACK DATABASE (Values per 100g) ──
    mock_db = {
        "apple": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2},
        "banana": {"calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3},
        "orange": {"calories": 47, "protein": 0.9, "carbs": 12, "fat": 0.1},
        "broccoli": {"calories": 34, "protein": 2.8, "carbs": 6.6, "fat": 0.4},
        "potato": {"calories": 77, "protein": 2.0, "carbs": 17, "fat": 0.1},
        "tomato": {"calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
        "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
        "beef": {"calories": 250, "protein": 26, "carbs": 0, "fat": 15},
        "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11},
        "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
        "pasta": {"calories": 131, "protein": 5.0, "carbs": 25, "fat": 1.1},
        "bread": {"calories": 265, "protein": 9.0, "carbs": 49, "fat": 3.2},
        "milk": {"calories": 42, "protein": 3.4, "carbs": 5.0, "fat": 1.0},
        "cheese": {"calories": 402, "protein": 25, "carbs": 1.3, "fat": 33},
        "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10},
        "burger": {"calories": 295, "protein": 17, "carbs": 24, "fat": 14},
    }

    # ── 2. TRY THE USDA API ──
    # 'DEMO_KEY' allows 30 requests per hour without an account.
    # Get a free permanent key at: https://fdc.nal.usda.gov/api-key-signup.html
    API_KEY = "DEMO_KEY"

    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": food_item,
        "api_key": API_KEY,
        "pageSize": 1 # We only need the top result
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if "foods" not in data or len(data["foods"]) == 0:
            raise ValueError(f"No results found in USDA API for {food_item}")

        food_data = data["foods"][0]
        nutrients = food_data.get("foodNutrients", [])

        # Initialize defaults
        cal_100g, carbs_100g, protein_100g, fat_100g = 0, 0, 0, 0

        # Loop through the USDA nutrient array to extract the macros
        for n in nutrients:
            name = n.get("nutrientName", "").lower()
            amount = n.get("value", 0)

            if "energy" in name and n.get("unitName", "").lower() == "kcal":
                cal_100g = amount
            elif "protein" in name:
                protein_100g = amount
            elif "carbohydrate, by difference" in name:
                carbs_100g = amount
            elif "total lipid (fat)" in name:
                fat_100g = amount

    # ── 3. IF API FAILS, USE MOCK DATA ──
    except Exception as e:
        print(f"[Nutrition Warning] USDA API bypassed or failed ({e}). Using offline database...")

        food_key = food_item.lower().strip()
        matched_key = next((k for k in mock_db.keys() if k in food_key), None)

        if matched_key:
            cal_100g = mock_db[matched_key]["calories"]
            carbs_100g = mock_db[matched_key]["carbs"]
            protein_100g = mock_db[matched_key]["protein"]
            fat_100g = mock_db[matched_key]["fat"]
        else:
            return {"error": f"I couldn't fetch live data, and '{food_item}' is not in my offline memory."}

    # ── 4. SCALE AND RETURN DATA ──
    factor = grams / 100.0

    return {
        "food": food_item,
        "grams": grams,
        "calories": cal_100g * factor,
        "carbs": carbs_100g * factor,
        "protein": protein_100g * factor,
        "fat": fat_100g * factor,
    }

# ---------------------------------------------------------------------------
# Kitchen control fulfillment functions
# ---------------------------------------------------------------------------

def ToggleLights(slots: dict, kitchen) -> dict:
    """
    Toggle or set kitchen lights.

    Slot keys:
        STATE (str, optional): "on" / "off" — defaults to toggling current state
    """
    state_raw = _unwrap(slots.get("STATE", ""))

    if state_raw:
        new_state = _parse_bool_slot(state_raw)
    else:
        # No explicit state — toggle current
        new_state = not kitchen.lights_on

    kitchen.set_lights(new_state)

    return {
        "intent":  "ToggleLights",
        "lights":  new_state,
        "message": f"Lights turned {'on' if new_state else 'off'}.",
        # UI update hint — ui.py reads this to update the lights_toggle checkbox
        "ui_updates": {
            "lights_toggle": new_state,
            "lights_status": "On" if new_state else "Off",
        },
    }


def SetOvenTemperature(slots: dict, kitchen: Kitchen) -> dict:
    temp_raw  = _unwrap(slots.get("TEMPERATURE", ""))
    unit_raw  = _unwrap(slots.get("UNIT", ""))
    mode_raw  = _unwrap(slots.get("COOK_MODE", ""))

    unit      = unit_raw.strip().upper() if unit_raw else "°F"
    cook_mode = mode_raw.strip().title() if mode_raw else "Bake"

    digits = re.sub(r'\D', '', temp_raw) if temp_raw else ""
    target_temp = int(digits) if digits else 0
    oven_on = target_temp > 0

    message = kitchen.set_oven(oven_on, target_temp)

    # Automatically opt the user in for a notification if it's heating up
    if oven_on and kitchen.oven_temp < target_temp:
        kitchen.notify_user_when_ready = True

    # FIX: Show the current temperature in the UI updates, not the target!
    current_temp = kitchen.oven_temp
    status_msg = (f"{cook_mode} (Heating: {current_temp}{unit} → {target_temp}{unit})"
                  if current_temp != target_temp and oven_on else
                  f"{cook_mode} at {target_temp}{unit} — {message}")

    return {
        "intent":      "SetOvenTemperature",
        "oven_on":     oven_on,
        "current_temp": current_temp,
        "target_temp": target_temp,
        "unit":        unit,
        "cook_mode":   cook_mode,
        "message":     message,
        "ui_updates": {
            "oven_switch":     oven_on,
            "oven_temp":       current_temp, # Now accurately reflects starting temp
            "oven_status_box": status_msg,
        },
    }

def NotifyOvenReady(slots: dict, kitchen: Kitchen) -> dict:
    """
    Handles the user explicitly asking "Let me know when the oven is ready."
    """
    if not kitchen.oven_on:
        return {"intent": "NotifyOvenReady", "message": "The oven is currently turned off."}

    if kitchen.oven_temp == kitchen.target_temp:
        return {"intent": "NotifyOvenReady", "message": f"The oven is already preheated to {kitchen.target_temp}."}

    # Set the flag to true
    kitchen.notify_user_when_ready = True

    return {
        "intent": "NotifyOvenReady",
        "message": f"I will notify you as soon as the oven reaches {kitchen.target_temp} degrees."
    }

def QueryNutrition(slots: dict, kitchen) -> dict:
    """
    Look up nutrition info for a food item.

    Slot keys:
        ITEM  (str, required): food item name
        GRAMS (int, optional): serving size in grams (default 100)
    """
    item  = _unwrap(slots.get("ITEM", "") or slots.get("FOOD", ""))
    grams = int(re.sub(r'\D', '', slots.get("GRAMS", 100)))

    if not item:
        return {"intent": "QueryNutrition", "error": "No food item specified."}

    result = kitchen.get_nutrition(item, grams=grams)

    if "error" in result:
        return {"intent": "QueryNutrition", **result}

    return {
        "intent":  "QueryNutrition",
        "food":    result.get("food"),
        "grams":   result.get("grams"),
        "calories": result.get("calories"),
        "protein":  result.get("protein"),
        "carbs":    result.get("carbs"),
        "fat":      result.get("fat"),
        "message": (
            f"{result['food']} ({result['grams']}g): "
            f"{result['calories']:.0f} kcal, "
            f"{result['protein']:.1f}g protein, "
            f"{result['carbs']:.1f}g carbs, "
            f"{result['fat']:.1f}g fat"
            if all(result.get(k) is not None for k in ("calories", "protein", "carbs", "fat"))
            else f"Partial data found for {result['food']}."
        ),
    }


def SetCaloricGoal(slots: dict, kitchen: Kitchen) -> dict:
    # Use CALORIE_VALUE to match your slot list
    cal_raw = _unwrap(slots.get("CALORIE_VALUE", slots.get("CALORIES", "0")))
    cal = int(re.sub(r'\D', '', str(cal_raw))) if cal_raw else 0

    if cal <= 0:
        return {"intent": "SetCaloricGoal", "error": "Please provide a valid caloric goal."}

    kitchen.caloric_goal = cal
    return {
        "intent":       "SetCaloricGoal",
        "caloric_goal": cal,
        "message":      f"Daily caloric goal set to {cal} kcal.",
        "ui_updates": {
            "caloric_goal_input": cal
        }
    }

def EditShoppingList(slots: dict, kitchen) -> dict:
    action = _unwrap(slots.get("LIST_ACTION", "add")).lower()
    item   = _unwrap(slots.get("FOOD_ITEM", ""))

    if not item:
        return {"intent": "EditShoppingList", "error": "No item specified."}

    if action in ("add", "adding"):
        # Fetch Nutrition Data (Default 100g)
        nut_info = get_nutrition(item, grams=100)
        cal = nut_info.get("calories", 0) if isinstance(nut_info, dict) and "error" not in nut_info else 0

        # Add to Kitchen state
        result = kitchen.add_shopping_item(item, cal)
        message = f"Added '{item}' ({cal:.0f} kcal) to the shopping list."

        # Let the LLM handle the warning naturally (No threads needed!)
        if kitchen.caloric_goal > 0 and kitchen.current_calories > kitchen.caloric_goal:
            warning_msg = f"Warning: This brings your total to {kitchen.current_calories:.0f} calories, which exceeds your limit of {kitchen.caloric_goal}. I suggest removing some items."
            message += " " + warning_msg

    elif action in ("remove", "delete", "removing"):
        result = kitchen.remove_shopping_item(item)
        message = f"Removed '{item}' from the shopping list."

    else:
        return {"intent": "EditShoppingList", "error": f"Unknown action '{action}'."}

    return {
        "intent":        "EditShoppingList",
        "action":        action,
        "item":          item,
        "shopping_list": result,
        "message":       message,
        "ui_updates": {
            "shopping_list_box": result,
            "current_calories_display": f"{kitchen.current_calories:.0f} kcal",
        },
    }


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Fulfilment.Helpers import _parse_duration_to_seconds, _seconds_to_hms

def Greeting(slots: dict) -> dict:
    return {"intent": "Greeting", "message": "User said hello."}

def Goodbye(slots: dict) -> dict:
    return {"intent": "Goodbye", "message": "User said goodbye."}

def OOS(slots: dict) -> dict:
    return {"intent": "OOS", "message": "Request is out of scope."}

def SetTimer(slots: dict, kitchen) -> dict:
    """
    Set a countdown timer.

    Slot keys:
        DURATION (str, required): natural language or numeric duration
        LABEL    (str, optional): what the timer is for
    """
    # 1. Extract the raw slot strings safely
    raw_duration = slots.get("DURATION", [""])[0] if isinstance(slots.get("DURATION"), list) else slots.get("DURATION", "")
    label = slots.get("LABEL", ["Timer"])[0] if isinstance(slots.get("LABEL"), list) else slots.get("LABEL", "Timer")

    # 2. Parse into seconds
    duration_seconds = _parse_duration_to_seconds(raw_duration)

    if duration_seconds <= 0:
        return {
            "intent": "SetTimer",
            "error": f"Could not parse a valid duration from '{raw_duration}'. Try '3 minutes' or '90 seconds'.",
        }

    # 3. Lock the kitchen state and apply the timer
    with kitchen._lock:
        kitchen.timer_remaining = duration_seconds
        kitchen.timer_label = label

    # 4. Format the display string
    display_str = f"⏱️ {label} — {_seconds_to_hms(duration_seconds)}"

    # 5. Return the JSON fulfillment
    return {
        "intent": "SetTimer",
        "duration": duration_seconds,
        "label": label,
        "message": f"{label} set for {_seconds_to_hms(duration_seconds)}.",
        "ui_updates": {
            "timer_duration": duration_seconds,
            "timer_display": display_str,
        },
    }
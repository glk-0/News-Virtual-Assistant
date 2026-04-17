import numpy as np
import gradio as gr
import os
import configparser                                                    
import torch
from Fulfilment.Helpers import _seconds_to_hms
from Fulfilment.KitchenModule import create_kitchen, get_nutrition
from Fulfilment.CommandProcessor import CommandProcessor
from Backend.VAState import VAState
from Backend.Verification import UserVerifier
from Backend.WakeWord import WakeWordDetector
from Backend.ASR import ASREngine
from Backend.Classifiers import IntentClassifier
from Backend.TTS import TTSEngine
from Backend.LanguageProcessing import NLGEngine, LocalTranslator

# System Initializations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Atlas] Using device: {DEVICE}")
tts_engine = TTSEngine()
_kitchen = create_kitchen(tts_engine=tts_engine)
processor = CommandProcessor(kitchen=_kitchen)
va_state = VAState()
verifier = UserVerifier(model_path="./ModelWeights/User_verification_model.pth")
wake_detector = WakeWordDetector(model_path="./ModelWeights/Hey_Atlas_WakeWord.pth")
asr = ASREngine(model_size="Medium")
intent_clf = IntentClassifier(model_dir="./ModelWeights/atlas_intent_slot_model")
translator = LocalTranslator()
config = configparser.ConfigParser()
config.read('.cfg')
keys = config['gemini']['api_keys']
keys = [k.strip() for k in keys.split(',')]
nlg_engine = NLGEngine(api_keys=keys)

def handle_audio_verification(audio, selected_user):
    if audio is None:
        return "🔒 Locked", "secondary", "⚠️ No audio recorded."

    # Gradio type="numpy" gives (sample_rate, int16_numpy_array)
    sample_rate, audio_data = audio
    # Pass int16 directly — verify_audio saves to wav via sf.write which handles int16 natively

    is_verified, username, score = verifier.verify_audio(audio_data, sample_rate)

    if is_verified:
        va_state.unlock(username or selected_user or "User")
        va_state.add_atlas_message(f"Atlas unlocked! Hello, {va_state.verified_user}.")
        return get_status_label(), "primary", f"✅ Verified as {va_state.verified_user}"
    else:
        return "🔒 Locked", "secondary", "❌ Voice not recognized. Try password bypass."


def handle_password_bypass(password):
    """
    Called when user types a password and clicks Unlock.
    Returns: (status_text, status_variant, info_text)
    """
    is_valid, _ = verifier.verify_password(password)
    if is_valid:
        va_state.unlock("Bypass User")
        va_state.add_atlas_message("Atlas unlocked via password. Hello!")
        return get_status_label(), "primary", "✅ Password accepted. VA unlocked."
    return "🔒 Locked", "secondary", "❌ Incorrect password."


# ── TAB 2: WAKE WORD ──────────────────────────────────────────

def handle_wake_audio(audio):
    """
    Takes a recorded audio clip, formats it to 16kHz mono float32,
    and runs the wake word detector.
    """
    if va_state.is_locked:
        return "🔒 Locked — authenticate first.", "—"

    if audio is None:
        return "⚠️ No audio recorded.", "—"

    sample_rate, audio_data = audio
    temp_wav_path = "temp_wake.wav"

    try:
        import soundfile as sf
        import torchaudio
        import torch
        import os

        # 1. Save to temp wav to let torchaudio handle formatting safely
        sf.write(temp_wav_path, audio_data, sample_rate)

        # 2. Load and format
        waveform, sr = torchaudio.load(temp_wav_path)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Convert to 1D float32 numpy array
        audio_window = waveform.squeeze(0).numpy()

        # 4. Clean up temp file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

        # 5. Process through the Wake Word model
        detected, prob = wake_detector.process_window(audio_window)
        prob_text = f"Wake prob: {prob:.3f}"

        if detected:
            va_state.wake_word_detected()
            va_state.add_atlas_message("Wake word detected! I'm listening...")
            return "🎙️ Wake word detected!", prob_text
        else:
            return "❌ Wake word not detected. Try again.", prob_text

    except Exception as e:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        return f"⚠️ Error processing audio: {e}", "—"


def handle_wake_text_bypass(text):
    """
    Allows typing 'Hey Atlas' as a wake word bypass.
    Returns: (wake_status_text)
    """
    if va_state.is_locked:
        return "🔒 Locked — authenticate first."
    if wake_detector.text_trigger(text):
        va_state.wake_word_detected()
        va_state.add_atlas_message("Wake word received (text). Listening...")
        return "🎙️ Wake word triggered via text!"
    return f"'{text}' is not the wake word. Try 'Hey Atlas'."


# ── TAB 3: COMMAND (ASR + Chat) ───────────────────────────────

def handle_audio_command(audio, language_choice):
    """
    Called when user records a command audio and submits.
    audio: filepath string from gr.Audio(type="filepath")
    language_choice: "English" or "French"
    Returns: (transcribed_text)
    """
    if audio is None:
        return ""
    lang_code = asr.SUPPORTED_LANGUAGES.get(language_choice, "en")
    transcription = asr.transcribe_file(audio, language=lang_code)
    return transcription


def handle_send_command(text_input, audio_input, language_choice, history, use_bypass, bypass_intent,hide_payload, *bypass_slots):
    ui_updates = {
        "lights_toggle": None,
        "lights_status": None,
        "oven_switch": None,
        "oven_temp": None,
        "oven_status_box": None,
        "shopping_list_box": None,
    }
    if va_state.is_locked:
        return history, text_input, audio_input, "🔒 Locked — authenticate first.", None, ui_updates

    # 1. Resolve the raw input FIRST (Text takes priority over Audio)
    command_text = text_input.strip() if text_input and text_input.strip() else None

    if not command_text and audio_input:
        lang_code = asr.SUPPORTED_LANGUAGES.get(language_choice, "en")
        command_text = asr.transcribe_file(audio_input, language=lang_code)

    try:
        if use_bypass and bypass_intent:
            filled_slots = {}
            for i, slot_name in enumerate(ALL_SLOTS):
                slot_val = bypass_slots[i]
                if slot_val:
                    filled_slots[slot_name] = [slot_val.strip()]

            prediction = {
                "intent": bypass_intent,
                "slots": filled_slots,
                "confidence": 1.0
            }

            if not command_text:
                command_text = f"[Manual Override] Intent: {bypass_intent}"

            va_state.add_user_message(command_text)

        else:
            # 2. Check if we actually got a command
            if not command_text:
                return history, "", None, get_status_label(), None, ui_updates

            # Add original French text to UI Chatbot
            va_state.add_user_message(command_text)

            # 3. TRANSLATE the final input if it's in French
            backend_command = command_text
            if language_choice == "French":
                backend_command = translator.translate_to_english(command_text)
                print(f"[Debug] Translated locally for BERT: '{command_text}' -> '{backend_command}'")

            # 4. PREDICT using the translated English string!
            prediction = intent_clf.predict(backend_command)

        # 2. Get API Fulfillment Response
        api_json = processor.process(prediction, language=language_choice)

        # ── Extract kitchen UI updates ─────────────────────────────────────
        ui_updates = api_json.get("ui_updates", {}) if isinstance(api_json, dict) else {}

        import time
        ui_updates["_ts"] = time.time()

        # 3. LLM Natural Language Generation
        nl_response = nlg_engine.generate_natural_response(
            user_message=command_text,
            api_json=api_json,
            language=language_choice
        )

        # 4. Generate TTS Audio
        audio_path = tts_engine.generate_audio(nl_response, language=language_choice)

        # 5. Format Chat Output
        intent = prediction["intent"]
        conf   = prediction.get("confidence", 1.0) * 100
        slots  = prediction["slots"]

        chat_msg = f"🧠 **Detected:** `{intent}` ({conf:.1f}%)\n"
        if slots:
            chat_msg += f"🧩 **Slots:** {slots}\n\n"
        else:
            chat_msg += "\n"

        chat_msg += f"🤖 **Atlas:** {nl_response}"

        if not hide_payload:
            chat_msg += f"\n\n*(API Payload: {api_json})*"

    except Exception as e:
        chat_msg   = f"⚠️ **System Error:** {str(e)}"
        audio_path = None
        ui_updates = {}

    va_state.add_atlas_message(chat_msg)
    va_state.command_received()

    return va_state.get_gradio_history(), "", None, get_status_label(), audio_path, ui_updates

# ── STATUS HELPERS ────────────────────────────────────────────
def get_status_label() -> str:
    """Returns a human-readable status string for the status badge."""
    if va_state.is_locked:
        return "🔒 Locked"
    if va_state.state == VAState.LISTENING_COMMAND:
        return "🎙️ Listening for Command"
    if va_state.state == VAState.LISTENING_WAKE:
        return "👂 Awake — Say 'Hey Atlas'"
    return f"✅ Unlocked — Hello, {va_state.verified_user}!"

def toggle_lights(state: bool):
    _kitchen.set_lights(bool(state))
    return "On" if _kitchen.lights_on else "Off"

def update_oven(switch_state: bool, target_temp: int):
    msg = _kitchen.set_oven(bool(switch_state), int(target_temp))
    return msg or "Oven command issued"

UI_CALORIE_CACHE = {}

def format_inventory_display(raw_list):
    """Takes 'bread, apple' and formats to 'Bread (265kcal), Apple (52kcal)'"""
    if not raw_list: return ""
    items = [x.strip() for x in raw_list.split(",") if x.strip()]
    formatted_items = []
    for i in items:
        cal = UI_CALORIE_CACHE.get(i.lower())
        if cal is not None:
            formatted_items.append(f"{i.capitalize()} ({cal:.0f}kcal)")
        else:
            formatted_items.append(i.capitalize())
    return ", ".join(formatted_items)

def add_to_list(item: str):
    audio_path = gr.update() # Default to no audio

    if not item:
        return gr.update(), f"{_kitchen.current_calories:.0f} kcal", audio_path

    nut_info = get_nutrition(item, grams=100)
    cal = nut_info.get("calories", 0) if isinstance(nut_info, dict) and "error" not in nut_info else 0

    UI_CALORIE_CACHE[item.lower()] = cal
    raw_list = _kitchen.add_shopping_item(item, cal)

    # NEW: Check if manual addition exceeds the limit and generate TTS!
    if _kitchen.caloric_goal > 0 and _kitchen.current_calories > _kitchen.caloric_goal:
        warning_msg = f"Warning: Adding {item} brings your total to {_kitchen.current_calories:.0f} calories, exceeding your limit of {_kitchen.caloric_goal}."
        print(f"🔊 TRIGGERING TTS: {warning_msg}")
        try:
            global tts_engine
            if tts_engine:
                audio_file = tts_engine.generate_audio(warning_msg)
                if audio_file:
                    audio_path = gr.update(value=audio_file)
        except Exception as e:
            print(f"[UI TTS Error]: {e}")

    return format_inventory_display(raw_list), f"{_kitchen.current_calories:.0f} kcal", audio_path

def remove_from_list(item: str):
        # Backend searches for the raw item perfectly
        raw_list = _kitchen.remove_shopping_item(item)
        return format_inventory_display(raw_list), f"{_kitchen.current_calories:.0f} kcal"


def sync_slider_to_switch(temp_value, current_switch):
    """
    If the slider is moved above 0, force the oven switch ON.
    Otherwise leave it unchanged.
    """
    if temp_value > 0 and not current_switch:
        return gr.update(value=True)   # turn switch ON
    return gr.update()                 # no change


def sync_switch_to_slider(switch_value, current_temp):
    """
    If the oven switch is turned OFF, force the slider to 0.
    """
    if not switch_value and current_temp != 0:
        return gr.update(value=0)      # reset slider to 0
    return gr.update()                 # no change


# ── 3. Intercept Voice Assistant updates ──
def apply_kitchen_updates(updates: dict):
    def _u(key):
        val = updates.get(key)
        # If the voice assistant modified the list, format it before displaying!
        if key == "shopping_list_box" and val is not None:
            val = format_inventory_display(val)
        return gr.update(value=val) if key in updates else gr.update()

    timer_update = gr.update(active=True) if "timer_duration" in updates and updates["timer_duration"] > 0 else gr.update()

    return (
        _u("lights_toggle"), _u("lights_status"),
        _u("oven_switch"), _u("oven_temp"), _u("oven_status_box"),
        _u("shopping_list_box"), timer_update, _u("timer_display"),
        _u("caloric_goal_input"), _u("current_calories_display")
    )

def kitchen_timer_tick():
    """Decrements timer and triggers TTS alert when done."""
    with _kitchen._lock:
        remaining = _kitchen.timer_remaining
        label     = _kitchen.timer_label

    if remaining <= 0:
        # Wrap string in gr.update
        return gr.update(active=False), gr.update(value="No timer set"), gr.update()

    remaining -= 1

    with _kitchen._lock:
        _kitchen.timer_remaining = remaining

    if remaining <= 0:
        # ── TIMER HIT ZERO: TRIGGER AUDIO ALERT ──
        audio_path = gr.update()
        # alert_msg = f"Your timer for {label} is done." if label else "Your timer is done."

        # Check if the label is generic/empty, otherwise read the custom label
        if not label or label.strip().lower() in ["","manual timer", "timer"]:
            alert_msg = "Your timer is done."
        else:
            alert_msg = f"Your timer for {label} is done."

        print(f"🔊 TRIGGERING TTS: {alert_msg}")
        try:
            global tts_engine
            if tts_engine:
                audio_file = tts_engine.generate_audio(alert_msg)
                if audio_file:
                    audio_path = gr.update(value=audio_file)
        except Exception as e:
            print(f"[Timer TTS Error]: {e}")

        # Wrap string in gr.update
        return gr.update(active=False), gr.update(value=f"✅ {label} — Done!"), audio_path

    # Wrap string in gr.update
    return gr.update(), gr.update(value=f"⏱️ {label} — {_seconds_to_hms(remaining)}"), gr.update()


def set_manual_timer(val, unit, label):
    """Bypasses the NLP pipeline to set a timer directly from UI."""
    multiplier = 1
    if unit == "Minutes": multiplier = 60
    elif unit == "Hours": multiplier = 3600

    duration = int(val * multiplier)
    display_label = label if label else "Manual Timer"

    with _kitchen._lock:
        _kitchen.timer_remaining = duration
        _kitchen.timer_label = display_label

    display = f"⏱️ {display_label} — {_seconds_to_hms(duration)}"

    return gr.update(active=True), gr.update(value=display)

def reset_manual_timer():
    """Immediately stops and clears the active timer."""
    with _kitchen._lock:
        _kitchen.timer_remaining = 0
        _kitchen.timer_label = ""

    # We must return active=False to stop the Gradio timer from ticking!
    return gr.update(active=False), "No timer set"

def update_caloric_goal(val):
    _kitchen.caloric_goal = int(val) if val else 0

# Strip the B- and I- prefixes to get the base slot names
ALL_SLOTS = [
    'DURATION', 'LOCATION', 'DATE', 'TOPIC', 'SOURCE', 'REGION',
    'TEAM', 'LEAGUE', 'SEASON', 'TEMPERATURE', 'TEMP_UNIT',
    'COOK_MODE', 'LIGHT_STATE', 'LIST_ACTION', 'FOOD_ITEM',
    'QUANTITY', 'QUANTITY_UNIT', 'CALORIE_VALUE', 'LABEL'
]

# Map intents to their valid slots (combining your required slots + logical optional ones)
INTENT_TO_SLOTS = {
    'OOS': [],
    'Greeting': [],
    'Goodbye': [],
    'SetTimer': ['DURATION', 'LABEL'],
    'GetWeather': ['LOCATION', 'DATE'],
    'GetTopHeadlines': ['REGION', 'TOPIC'],
    'GetTopicNews': ['TOPIC'],
    'GetPublisherHeadlines': ['SOURCE'],
    'GetRegionNews': ['REGION'],
    'GetGameScore': ['TEAM', 'LEAGUE', 'DATE'],
    'GetTeamStanding': ['TEAM', 'LEAGUE'],
    'GetLeagueSchedule': ['LEAGUE', 'SEASON'],
    'SetOvenTemperature': ['TEMPERATURE', 'TEMP_UNIT'],
    'ToggleLights': ['LIGHT_STATE', 'LOCATION'],
    'EditShoppingList': ['LIST_ACTION', 'FOOD_ITEM', 'QUANTITY', 'QUANTITY_UNIT'],
    'QueryNutrition': ['FOOD_ITEM', 'QUANTITY', 'QUANTITY_UNIT'],
    'SetCaloricGoal': ['CALORIE_VALUE']
    #'NotifyOvenReady': ['FOOD_ITEM']
}

# ============================================================
# CELL 6 — GRADIO UI LAYOUT (Single-Page App + Instructions)
# ============================================================

# 1. Define a minimalist dark theme
atlas_theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "system-ui", "sans-serif"]
).set(
    body_background_fill="transparent",
    body_text_color="#f8fafc",
    block_background_fill="rgba(24, 24, 27, 0.65)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.1)",
    block_shadow="0 4px 30px rgba(0, 0, 0, 0.3)",
    button_primary_background_fill="#ffffff",
    button_primary_background_fill_hover="#e4e4e7",
    button_primary_text_color="#000000",
    button_secondary_background_fill="rgba(255, 255, 255, 0.1)",
    button_secondary_background_fill_hover="rgba(255, 255, 255, 0.2)",
    button_secondary_text_color="#ffffff",
    input_background_fill="rgba(0, 0, 0, 0.5)",
    input_border_color="rgba(255, 255, 255, 0.15)"
)

# 2. Advanced Custom CSS
with open("./UI/main.css", "r", encoding="utf-8") as f:
    custom_css = f.read()
PUBLISHER_OPTIONS = [
    "bbc", "cnn", "fox", "fox news", "new york times", "washington post",
    "reuters", "associated press", "the guardian",
    "espn", "sky sports", "nbc news", "abc news",
    "techcrunch", "the verge", "wired",
]

with gr.Blocks(theme=atlas_theme, css=custom_css, title="Atlas VA") as demo:

    # Global States
    audio_buffer_state = gr.State(np.array([], dtype=np.float32))
    kitchen_update_state = gr.State({})

    # ==========================================================
    # PAGE 0: LANDING PAGE
    # ==========================================================
    with gr.Column(visible=True, elem_id="landing_page") as landing_page:
        gr.Markdown("# Atlas Morning Virtual Assistant", elem_classes="landing-title")
        gr.Markdown("Build a smarter environment. Ask about sports, get the latest news, or control your smart kitchen by simply chatting with AI.", elem_classes="landing-subtitle")

        # Typewriter Input Box
        gr.HTML("""
        <div class="typewriter-container">
            <span class="typewriter-inner"></span>
        </div>
        """)

        start_btn = gr.Button("Start your day now!", variant="primary", elem_id="start_button")

    # ==========================================================
    # APP CONTAINER
    # ==========================================================
    with gr.Column(visible=False, elem_classes="animate-slide-up") as app_container:

        gr.Markdown("## Atlas", elem_classes="atlas-header")

        with gr.Row(elem_classes="glass-panel"):
            status_badge = gr.Button("🔒 Locked", variant="secondary", interactive=False, scale=2)
            status_info = gr.Textbox("Awaiting authentication protocol.", show_label=False, interactive=False, scale=8)

        # ------------------------------------------------------
        # STEP 1: VERIFICATION PAGE
        # ------------------------------------------------------
        with gr.Column(visible=True, elem_classes="animate-slide-up") as step_1_container:
            gr.Markdown("### Step 1: Verify Identity")

            # --- NEW INSTRUCTIONS ---
            with gr.Accordion("📖 Guide & Instructions", open=False):
                gr.Markdown("""
                **Welcome to Atlas User Verification!**
                * **Authorized Users:** Only **Ghali**, **Coralie**, and **Connor** are permitted to unlock this system via voice. Other people speaking will not be granted access.
                * **How to Authenticate:** Our voice recognition is text-independent. You do not need a specific passphrase, just speak naturally into the microphone for 1-3 seconds.
                * **Emergency Override:** If the voice model is failing or unavailable, you can bypass this step using the password: `CSI5180`.
                """)

            with gr.Row(elem_classes="glass-panel"):
                with gr.Column(scale=1):
                    verify_audio = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Voice Authentication")
                    verify_btn = gr.Button("Authenticate", variant="primary")
                with gr.Column(scale=1):
                    password_input = gr.Textbox(label="Emergency Bypass Password", type="password", placeholder="Enter key")
                    bypass_btn = gr.Button("Unlock", variant="secondary")

        # ------------------------------------------------------
        # STEP 2: WAKE WORD PAGE
        # ------------------------------------------------------
        with gr.Column(visible=False, elem_classes="animate-slide-up") as step_2_container:
            gr.Markdown("### Step 2: Wake up System")

            # --- NEW INSTRUCTIONS ---
            with gr.Accordion("📖 Guide & Instructions", open=False):
                gr.Markdown("""
                **Waking up the Assistant**
                * **The Wake Word:** Simply say **"Hey Atlas"** to activate the assistant.
                * **Who can speak?** At this stage, the system assumes the verified user is speaking. Anyone speaking the wake word will successfully trigger the system.
                * **Test the Limits:** We encourage you to test near-misses! Try saying things like *"Hey at last"* or *"Hey at least"* to see how robust the model is to false positives.
                """)

            with gr.Row(elem_classes="glass-panel"):
                with gr.Column(scale=2):
                    wake_audio = gr.Audio(sources=["microphone"], type="numpy", label="🎙️ Say 'Hey Atlas'")
                    wake_audio_btn = gr.Button("Wake up Atlas", variant="primary")
                    with gr.Row():
                        wake_status = gr.Textbox(label="Status", value="Standby", interactive=False)
                        wake_prob = gr.Textbox(label="Confidence", value="—", interactive=False)
                with gr.Column(scale=1):
                    wake_text_input = gr.Textbox(placeholder="> Hey Atlas", label="Text Override", show_label=False)
                    wake_text_btn = gr.Button("Inject Wake Word", variant="secondary")
                    wake_text_result = gr.Textbox(label="Log", interactive=False)

        # ------------------------------------------------------
        # STEP 3: MAIN DASHBOARD
        # ------------------------------------------------------
        with gr.Column(visible=False, elem_classes="animate-slide-up") as step_3_container:
            with gr.Tabs():
                # TAB A: Command Center
                with gr.Tab("💬 Command Center"):

                    # --- NEW INSTRUCTIONS ---
                    with gr.Accordion("📖 Guide & Supported Commands", open=False):
                        gr.Markdown("""
                        **Using the Command Center**
                        * **Bilingual Support:** You can type or speak your commands in either **English** or **French**. Make sure you select the right language!
                        * **Transcription Correction:** If you record an audio command, you can click "Transcribe" to convert it to text, correct any errors in the text box manually, and then hit "Transmit".
                        * **Manual Intent Override:** If the natural language model fails to detect your intent or misses specific details (slots), open the **🛠️ Debug** menu. You can manually force an intent and provide the exact parameters.

                        **Supported Capabilities (Intents & Slots)**

                        | Intent | Description | Relevant Slots | Sample Sentence |
                        | :--- | :--- | :--- | :--- |
                        | **Greeting / Goodbye** | Standard conversational greetings. | *None* | "Good morning, Atlas!" / "Goodbye." |
                        | **SetTimer** | Set a countdown timer. | `DURATION` (M) | "Set a timer for 15 minutes." |
                        | **GetWeather** | Check the current or future weather. | `LOCATION` (O), `DATE` (O) | "What's the weather like in Ottawa tomorrow?" |
                        | **GetTopHeadlines** | Fetch the latest top news headlines. | `REGION` (O), `TOPIC` (O) | "Give me the top news headlines." |
                        | **GetTopicNews** | Search the news for a specific subject. | `TOPIC` (M) | "Show me the latest news about technology." |
                        | **GetPublisherHeadlines**| Get news from a specific media outlet. | `SOURCE` (M) | "Get the latest news from BBC." |
                        | **GetRegionNews** | Fetch news localized to a specific area. | `REGION` (M) | "What is the news in Canada today?" |
                        | **GetGameScore** | Check the score of a sports match. | `TEAM` (M), `LEAGUE` (O), `DATE` (O) | "What was the score of the Habs game?" |
                        | **GetTeamStanding** | Check a team's ranking in their league. | `TEAM` (M), `LEAGUE` (O) | "Where does Arsenal stand right now?" |
                        | **GetLeagueSchedule** | Look up upcoming sports schedules. | `LEAGUE` (M), `SEASON` (O) | "What is the NBA schedule?" |
                        | **SetOvenTemperature** | Power on the oven and set the heat. | `TEMPERATURE` (M), `TEMP_UNIT` (O), `COOK_MODE` (O) | "Preheat the oven to 400 degrees Fahrenheit." |
                        | **ToggleLights** | Turn the kitchen lights on or off. | `LIGHT_STATE` (M), `LOCATION` (O) | "Turn off the kitchen lights." |
                        | **EditShoppingList** | Add or remove items from your grocery list. | `LIST_ACTION` (M), `FOOD_ITEM` (M), `QUANTITY` (O), `QUANTITY_UNIT` (O) | "Add 3 pounds of apples to my shopping list." |
                        | **QueryNutrition** | Look up calorie info for a specific food. | `FOOD_ITEM` (M), `QUANTITY` (O), `QUANTITY_UNIT` (O) | "How many calories are in a banana?" |
                        | **SetCaloricGoal** | Set your daily calorie limit. | `CALORIE_VALUE` (M) | "Set my daily caloric limit to 2500." |
                        | **OOS** | Out of Scope (requests Atlas cannot handle). | *None* | "Can you book me a flight to Paris?" |

                        *(Note: **M** = Mandatory slot required for the command to execute properly, **O** = Optional slot that adds context but isn't strictly required).*
                        """)

                    with gr.Row(elem_classes="glass-panel"):
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(label="System Log", height=450, type="tuples", bubble_full_width=False)
                            with gr.Row():
                                cmd_text = gr.Textbox(placeholder="Ask Atlas a question...", show_label=False, scale=4)
                                cmd_audio = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Voice Input", scale=1)
                            with gr.Row():
                                transcribe_btn = gr.Button("Transcribe", variant="secondary")
                                send_btn = gr.Button("Transmit", variant="primary")
                        with gr.Column(scale=1):
                            atlas_voice = gr.Audio(label="🔊 Synthesizer Output", autoplay=True, interactive=False)
                            language_selector = gr.Radio(choices=["English", "French"], value="English", label="Language")
                            session_state_display = gr.Textbox(label="State", value="LISTENING", interactive=False)
                            with gr.Accordion("🛠️ Debug", open=False):
                                hide_payload = gr.Checkbox(label="Hide API Payload", value=False)
                                use_manual_bypass = gr.Checkbox(label="Force Intent")
                                manual_intent = gr.Dropdown(choices=list(INTENT_TO_SLOTS.keys()), label="Target Intent")
                                slot_inputs = {}
                                for slot_name in ALL_SLOTS:
                                    if slot_name == "SOURCE":
                                        slot_inputs[slot_name] = gr.Dropdown(
                                            choices=PUBLISHER_OPTIONS,
                                            label=slot_name,
                                            visible=False,
                                        )
                                    else:
                                        slot_inputs[slot_name] = gr.Textbox(label=slot_name, visible=False)

                # TAB B: Kitchen Hardware
                with gr.Tab("🍳 Smart Kitchen"):

                    # --- NEW INSTRUCTIONS ---
                    with gr.Accordion("📖 Guide & Instructions", open=False):
                        gr.Markdown("""
                        **Kitchen Control Dashboard**

                        This panel provides a live view of your smart kitchen. You have **Dual Control**: everything here can be manipulated via voice commands in the Command Center, OR manually bypassed by clicking the UI elements below.

                        * **💡 Environment:** Manually flip the kitchen lights or set quick custom timers without talking to Atlas. Atlas will let you know wwhen the timer is over.
                        * **🔥 Thermal Unit (Oven):** Turn the oven on/off and slide to your desired temperature. The live telemetry will update to show the current heat state. Atlas will let you know the oven is ready.
                        * **🛒 Inventory & Nutrition:** Add grocery items to your shopping list. Atlas connects to external APIs to automatically calculate the nutritional value (calories) of items you add and tracks them against your daily limit! Atlas will let you know when you exceed your daily limit.
                        """)

                    with gr.Row():
                        # Environment
                        with gr.Column(elem_classes="glass-panel"):
                            gr.Markdown("#### 💡 Environment")
                            lights_toggle = gr.Checkbox(label="Main Lights", value=False)
                            lights_status = gr.Textbox(label="Status", interactive=False)
                            gr.Markdown("<hr>")
                            timer_display = gr.Textbox(label="⏱️ Timer", value="No timer", interactive=False)
                            general_timer = gr.Timer(value=1, active=False)
                            with gr.Accordion("Manual Timer", open=False):
                                manual_timer_val = gr.Number(label="Value", value=1)
                                manual_timer_unit = gr.Dropdown(choices=["Seconds", "Minutes", "Hours"], value="Minutes", show_label=False)
                                manual_timer_label = gr.Textbox(placeholder="Label")
                                manual_timer_btn = gr.Button("Start", variant="primary")
                                manual_timer_reset_btn = gr.Button("Abort", variant="stop")

                        # Oven
                        with gr.Column(elem_classes="glass-panel"):
                            gr.Markdown("#### 🔥 Oven")
                            oven_switch = gr.Checkbox(label="Main Power", value=False)
                            oven_temp = gr.Slider(0, 500, value=0, step=5, label="Temp (°F)")
                            oven_status_box = gr.Textbox(label="State", interactive=False)
                            current_temp_display = gr.Textbox(label="Live Temp", value="0°F", interactive=False)
                            temp_timer = gr.Timer(value=0.5)

                        # Inventory
                        with gr.Column(elem_classes="glass-panel"):
                            gr.Markdown("#### 🛒 Inventory & Nutrition")

                            shopping_list_box = gr.Textbox(
                                label="Active Manifest",
                                placeholder="List is empty.",
                                interactive=False,
                                lines=4
                            )

                            # Grouped the input and buttons into one sleek line
                            with gr.Row():
                                shopping_item = gr.Textbox(
                                    placeholder="Enter item (e.g., Bread)...",
                                    show_label=False,
                                    scale=3
                                )
                                add_btn = gr.Button("Add (+)", variant="primary", scale=1)
                                remove_btn = gr.Button("Drop (-)", variant="stop", scale=1)

                            gr.Markdown("<hr style='border-color: rgba(255, 255, 255, 0.1); margin: 10px 0;'>")

                            # Grouped the nutrition tracking side-by-side
                            with gr.Row():
                                caloric_goal_input = gr.Number(label="🎯 Daily Caloric Cap", value=0)
                                current_calories_display = gr.Textbox(label="🔥 Current Calorie Count", value="0 kcal", interactive=False)

    # ==========================================================
    # ROUTING & EVENT BINDINGS
    # ==========================================================

    start_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=None,
        outputs=[landing_page, app_container]
    )

    def process_auth_and_route(audio, name):
        badge, badge2, info = handle_audio_verification(audio, name)
        if "✅" in info:
            return badge, badge2, info, gr.update(visible=False), gr.update(visible=True)
        return badge, badge2, info, gr.update(visible=True), gr.update(visible=False)

    verify_btn.click(
        fn=process_auth_and_route,
        inputs=[verify_audio, gr.State(None)],
        outputs=[status_badge, status_badge, status_info, step_1_container, step_2_container]
    )

    def process_bypass_and_route(pwd):
        badge, badge2, info = handle_password_bypass(pwd)
        if "✅" in info:
            return badge, badge2, info, gr.update(visible=False), gr.update(visible=True)
        return badge, badge2, info, gr.update(visible=True), gr.update(visible=False)

    bypass_btn.click(
        fn=process_bypass_and_route,
        inputs=[password_input],
        outputs=[status_badge, status_badge, status_info, step_1_container, step_2_container]
    )

    def process_wake_and_route(audio):
        status, prob = handle_wake_audio(audio)
        if "🎙️" in status:
            return status, prob, gr.update(visible=False), gr.update(visible=True)
        return status, prob, gr.update(visible=True), gr.update(visible=False)

    wake_audio_btn.click(
        fn=process_wake_and_route,
        inputs=[wake_audio],
        outputs=[wake_status, wake_prob, step_2_container, step_3_container]
    )

    def process_wake_text_and_route(text):
        status = handle_wake_text_bypass(text)
        if "🎙️" in status:
            return status, gr.update(visible=False), gr.update(visible=True)
        return status, gr.update(visible=True), gr.update(visible=False)

    wake_text_btn.click(
        fn=process_wake_text_and_route,
        inputs=[wake_text_input],
        outputs=[wake_text_result, step_2_container, step_3_container]
    )

    def update_slot_visibility(intent):
      if not intent:
          updates = []
          for slot in ALL_SLOTS:
              if slot == "SOURCE":
                  updates.append(gr.update(visible=False, value=None))
              else:
                  updates.append(gr.update(visible=False, value=""))
          return updates

      valid_slots = INTENT_TO_SLOTS.get(intent, [])
      updates = []
      for slot in ALL_SLOTS:
          if slot in valid_slots:
              updates.append(gr.update(visible=True))
          else:
              if slot == "SOURCE":
                  updates.append(gr.update(visible=False, value=None))
              else:
                  updates.append(gr.update(visible=False, value=""))
      return updates

    manual_intent.change(fn=update_slot_visibility, inputs=[manual_intent], outputs=list(slot_inputs.values()))
    transcribe_btn.click(fn=handle_audio_command, inputs=[cmd_audio, language_selector], outputs=[cmd_text])

    command_inputs = [cmd_text, cmd_audio, language_selector, chatbot, use_manual_bypass, manual_intent, hide_payload] + list(slot_inputs.values())
    command_outputs = [chatbot, cmd_text, cmd_audio, status_info, atlas_voice, kitchen_update_state]

    send_btn.click(fn=handle_send_command, inputs=command_inputs, outputs=command_outputs)
    cmd_text.submit(fn=handle_send_command, inputs=command_inputs, outputs=command_outputs)

    lights_toggle.change(fn=toggle_lights, inputs=[lights_toggle], outputs=[lights_status])
    oven_temp.change(fn=sync_slider_to_switch, inputs=[oven_temp, oven_switch], outputs=[oven_switch])
    oven_switch.change(fn=sync_switch_to_slider, inputs=[oven_switch, oven_temp], outputs=[oven_temp])
    oven_switch.change(fn=update_oven, inputs=[oven_switch, oven_temp], outputs=[oven_status_box])
    oven_temp.change(fn=update_oven, inputs=[oven_switch, oven_temp], outputs=[oven_status_box])

    def poll_oven_temp():
      audio_to_play = gr.update()
      with _kitchen._lock:
          if _kitchen.ready_audio_path:
              audio_to_play = _kitchen.ready_audio_path
              _kitchen.ready_audio_path = None
      return f"{int(_kitchen.oven_temp)}°F", audio_to_play

    temp_timer.tick(fn=poll_oven_temp, inputs=[], outputs=[current_temp_display, atlas_voice])
    caloric_goal_input.change(fn=update_caloric_goal, inputs=[caloric_goal_input], outputs=[])

    add_btn.click(fn=add_to_list, inputs=[shopping_item], outputs=[shopping_list_box, current_calories_display, atlas_voice])
    remove_btn.click(fn=remove_from_list, inputs=[shopping_item], outputs=[shopping_list_box, current_calories_display])

    kitchen_update_state.change(
        fn=apply_kitchen_updates,
        inputs=[kitchen_update_state],
        outputs=[lights_toggle, lights_status, oven_switch, oven_temp, oven_status_box, shopping_list_box, general_timer, timer_display, caloric_goal_input, current_calories_display],
    )

    manual_timer_btn.click(fn=set_manual_timer, inputs=[manual_timer_val, manual_timer_unit, manual_timer_label], outputs=[general_timer, timer_display])
    manual_timer_reset_btn.click(fn=reset_manual_timer, inputs=[], outputs=[general_timer, timer_display])
    general_timer.tick(fn=kitchen_timer_tick, inputs=[], outputs=[general_timer, timer_display, atlas_voice])

demo.launch(debug=True, share=True)
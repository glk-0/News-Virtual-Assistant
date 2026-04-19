# 🌅 Atlas Morning Virtual Assistant

**Atlas** is a specialized morning virtual assistant designed to simplify daily routines through natural voice interaction like Siri, or Alexa. Developed as the final project for **CSI 5180: Topics in AI - Virtual Assistants (Winter 2026)**, Atlas acts as an interactive command center to help users multitask during their busy mornings.


https://github.com/user-attachments/assets/f02dc88e-9ca2-4dce-b084-e5e6534ccbc8


---

## ✨ Key Features
* **News & Information:** Get top headlines or search for specific news topics.
* **Sports Updates:** Check live scores, schedules, and team updates.
* **Smart Kitchen Control:** Control oven temperatures, toggle kitchen lights, manage a shopping list, and track daily caloric intake.
* **Bilingual Interaction:** Seamlessly switch between English and French for both speech recognition and text-to-speech output.
* **Gradio Dashboard:** A clean, intuitive web-based command center to view logs, system statuses, and interactive elements.

## 👥 Target Audience
* Busy individuals and students multitasking in the morning.
* Users who prefer hands-free, voice-first interaction while cooking or getting ready.

---

## 🧠 System Architecture & AI Pipeline

Atlas is built on a modular pipeline of state-of-the-art AI models, connecting from the moment you speak to the moment the assistant replies. All steps include a bypass in case the module fails.

1.  **User Identification (Speaker Verification)**
    * A custom voice biometric system ensures secure access. The system is locked and only authorizes registered users (Ghali, Connor, Coralie) to access the assistant. Bypass : Enter password 'CSI5180'.
2.  **Wake Word Detection**
    * Powered by a fine tuned **openWakeWord**, the system lets you record the trigger phrase *"Hey Atlas"* to wake the assistant. Bypass : Type the wakeword 'Hey Atlas'.
3.  **Automatic Speech Recognition (ASR)**
    * Audio is transcribed using OpenAI's **Whisper (Medium)** model. This provides highly robust, text-independent transcription in both English and French. Bypass : The user can edit the text transcription before moving to the next step.
4.  **Intent Detection & Slot Filling**
    * A fine-tuned **BERT** model analyzes the transcribed text to classify the user's intent (e.g., `SetOvenTemperature`, `GetNews`) and extracts relevant entities/slots (e.g., temperature, food item, sports team). Bypass : The user can manually select an intent and values for the slots.
5.  **Fulfillment & API Integration**
    * **News:** Integrates `GNews` and `NewsAPI` to fetch real-time global and local headlines.
    * **Sports:** Uses the `ESPN API` for up-to-date sports data.
    * **Kitchen & Nutrition:** Queries the `USDA FoodData Central API` (with an offline fallback database) to calculate calories for shopping list items.
6.  **Natural Language Generation (NLG)**
    * Fulfillment data is passed to **Google Gemini** (LLM), which is guided by prompt templates to generate a conversational, natural-sounding response rather than a robotic data readout. Bypass : The system fallback to a set of harcoded templates for each intent, that are more rigid, but avoid system failure.
7.  **Text-to-Speech (TTS)**
    * The generated text is converted back to audio using the **Kokoro** TTS model, which supports highly natural, expressive bilingual voice generation.
8.  **User Interface**
    * The entire pipeline is wrapped in a **Gradio** web application, providing visual feedback, text overrides, bypass passwords, and interactive UI components for the smart kitchen.

### Supported Capabilities (Intents & Slots)

| Intent | Description | Relevant Slots | Sample Sentence |
| :--- | :--- | :--- | :--- |
| **Greeting / Goodbye** | Standard conversational greetings. | None | "Good morning, Atlas!" / "Goodbye." |
| **SetTimer** | Set a countdown timer. | DURATION (M) | "Set a timer for 15 minutes." |
| **GetWeather** | Check the current or future weather. | LOCATION (O), DATE (O) | "What's the weather like in Ottawa tomorrow?" |
| **GetTopHeadlines** | Fetch the latest top news headlines. | REGION (O), TOPIC (O) | "Give me the top news headlines." |
| **GetTopicNews** | Search the news for a specific subject. | TOPIC (M) | "Show me the latest news about technology." |
| **GetPublisherHeadlines** | Get news from a specific media outlet. | SOURCE (M) | "Get the latest news from BBC." |
| **GetRegionNews** | Fetch news localized to a specific area. | REGION (M) | "What is the news in Canada today?" |
| **GetGameScore** | Check the score of a sports match. | TEAM (M), LEAGUE (O), DATE (O) | "What was the score of the Habs game?" |
| **GetTeamStanding** | Check a team's ranking in their league. | TEAM (M), LEAGUE (O) | "Where does Arsenal stand right now?" |
| **GetLeagueSchedule** | Look up upcoming sports schedules. | LEAGUE (M), SEASON (O) | "What is the NBA schedule?" |
| **SetOvenTemperature** | Power on the oven and set the heat. | TEMPERATURE (M), TEMP_UNIT (O), COOK_MODE (O) | "Preheat the oven to 400 degrees Fahrenheit." |
| **ToggleLights** | Turn the kitchen lights on or off. | LIGHT_STATE (M), LOCATION (O) | "Turn off the kitchen lights." |
| **EditShoppingList** | Add or remove items from your grocery list. | LIST_ACTION (M), FOOD_ITEM (M), QUANTITY (O), QUANTITY_UNIT (O) | "Add 3 pounds of apples to my shopping list." |
| **QueryNutrition** | Look up calorie info for a specific food. | FOOD_ITEM (M), QUANTITY (O), QUANTITY_UNIT (O) | "How many calories are in a banana?" |
| **SetCaloricGoal** | Set your daily calorie limit. | CALORIE_VALUE (M) | "Set my daily caloric limit to 2500." |
| **OOS** | Out of Scope (requests Atlas cannot handle). | None | "Can you book me a flight to Paris?" |

*(Note: **M** = Mandatory slot required for the command to execute properly, **O** = Optional slot that adds context but isn't strictly required).*
---

## ⚙️ Setup and Installation

### 1. `.cfg` File Setup
Mirror the following format with your API keys in a `.cfg` file in the root directory. 
If you have multiple keys for each API, comma separate them (ex. `api_keys = key1, key2, key3`):

```ini
[newsapi]
api_keys = key

[gnews]
api_keys = key

[gemini]
api_keys = key
## .cfg File Setup
Mirror the following format with your api keys.

If you have multiple keys for each api comma separate them: 

ex. api_keys = key1, key2, key3
    
    [newsapi]
    api_keys = key

    [gnews]
    api_keys = key

    [gemini]
    api_keys = key 
```
### 2. Install Python Dependencies
Install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```


### 3. Install eSpeak (Required for TTS)
The TTS System (Kokoro) requires `espeak-ng` to be installed on your machine. Use the appropriate method for your operating system below:

#### Windows
1. Download the latest `eSpeak-NG` installer from the official releases page.
2. Run the installer and follow the prompts.
3. After installation, verify it works by opening a terminal and running:
   ```bash
   espeak-ng --version
   ```
   *(If the command is not found, ensure you add the installation directory to your system PATH).*

#### macOS
The easiest method is using Homebrew. If you do not have Homebrew installed, visit the Homebrew website for installation instructions.
```bash
brew install espeak-ng
```
Verify installation:
```bash
espeak-ng --version
```

#### Linux

**Debian/Ubuntu:**
```bash
sudo apt update
sudo apt install espeak-ng
```

**Fedora:**
```bash
sudo dnf install espeak-ng
```

**Arch Linux:**
```bash
sudo pacman -S espeak-ng
```

Verify installation:
```bash
espeak-ng --version
```
### Link to the model weights : https://drive.google.com/drive/folders/1LsAN32TMG9HX-lgIBH8NGgM8EnLuCIhF?usp=drive_link
---


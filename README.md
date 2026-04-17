# News-Virtual-Assistant
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

## Pip install the requirements list
    pip install -r requirements.txt

## Espeak
TTS System requires Espeak install use one of the following to install:

### Windows

    Download the latest eSpeak‑NG installer from the official releases page.
    
    Run the installer and follow the prompts.
    
    After installation, verify it works:
    
    espeak-ng --version
    
    If the command is not found, add the installation directory to your system PATH.
        
### macOS
    
    The easiest method is using Homebrew:
    
    brew install espeak-ng
    
    Verify installation:
    
    espeak-ng --version
    
    If you do not have Homebrew installed, visit the Homebrew website for installation instructions.
    
### Linux
    
#### Debian/Ubuntu
    
    sudo apt update
    sudo apt install espeak-ng
    
#### Fedora
    
    sudo dnf install espeak-ng
    
#### Arch Linux
    
    sudo pacman -S espeak-ng
    
### Verify installation:
        
    espeak-ng --version


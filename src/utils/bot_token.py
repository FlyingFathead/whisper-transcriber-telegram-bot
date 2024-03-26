# ~~~ read the telegram bot token ~~~
import os
import sys
import configparser

def get_bot_token():
    config = configparser.ConfigParser()
    # Adjust the path relative to the location of the script in `src`
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini')
    config.read(config_path)

    prefer_env = config.getboolean('DEFAULT', 'PreferEnvForBotToken', fallback=True)

    if prefer_env:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if bot_token is not None:
            return bot_token

    try:
        # Adjust the path for `bot_token.txt` as well, assuming it's in the same directory as `config.ini`
        token_file_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'bot_token.txt')
        with open(token_file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        if not prefer_env:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            if bot_token is not None:
                return bot_token

        print("The TELEGRAM_BOT_TOKEN environment variable is not set, and `bot_token.txt` was not found. Please set either one and adjust `config.ini` if needed for the preferred load order.")
        sys.exit(1)
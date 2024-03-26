# ~~~ read the telegram bot token ~~~

import os
import configparser
import sys

def get_bot_token():
    # Correctly ascend two levels to get the project root from bot_token.py in src/utils
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config', 'config.ini')
    token_file_path = os.path.join(base_dir, 'config', 'bot_token.txt')

    print(f"Debug: Base directory is {base_dir}")
    print(f"Debug: Config path is {config_path}")
    print(f"Debug: Token file path is {token_file_path}")

    # Check if the paths actually exist
    if not os.path.exists(config_path):
        print("Error: config.ini not found at the expected path.")
        sys.exit(1)

    if not os.path.exists(token_file_path) and not os.getenv('TELEGRAM_BOT_TOKEN'):
        print("Error: bot_token.txt not found at the expected path and TELEGRAM_BOT_TOKEN environment variable is not set.")
        sys.exit(1)

    # Reading the config
    config = configparser.ConfigParser()
    config.read(config_path)
    prefer_env = config.getboolean('DEFAULT', 'PreferEnvForBotToken', fallback=True)

    if prefer_env and os.getenv('TELEGRAM_BOT_TOKEN'):
        return os.getenv('TELEGRAM_BOT_TOKEN')

    # Try to read the token from the file if the environment variable isn't preferred or set
    try:
        with open(token_file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Error: Failed to read bot_token.txt.")
        sys.exit(1)

    # Fallback error message
    print("The bot token could not be determined.")
    sys.exit(1)
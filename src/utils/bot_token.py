# ~~~ Enhanced Read Telegram Bot Token with Configurable Fallback, Appropriate Logging, and Validity Check ~~~

import os
import configparser
import logging
from pathlib import Path
import sys

# Set up basic logging configuration
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

class BotTokenError(Exception):
    """Custom exception for bot token retrieval failures."""
    pass

def get_bot_token():
    try:
        # Ascend two levels to get the project root from bot_token.py in src/utils
        base_dir = Path(__file__).resolve().parents[2]
        config_path = base_dir / 'config' / 'config.ini'
        token_file_path = base_dir / 'config' / 'bot_token.txt'

        logging.debug(f"Base directory: {base_dir}")
        logging.debug(f"Config path: {config_path}")
        logging.debug(f"Token file path: {token_file_path}")

        # Verify config.ini exists
        if not config_path.is_file():
            raise BotTokenError(f"config.ini not found at {config_path}.")

        # Read configuration
        config = configparser.ConfigParser()
        config.read(config_path)

        # Validate configuration
        if 'DEFAULT' not in config:
            raise BotTokenError("Missing 'DEFAULT' section in config.ini.")

        prefer_env = config.getboolean('DEFAULT', 'PreferEnvForBotToken', fallback=True)
        allow_fallback = config.getboolean('DEFAULT', 'AllowFallback', fallback=True)
        ask_for_token = config.getboolean('DEFAULT', 'AskForTokenIfNotFound', fallback=True)

        invalid_tokens = [
            'YourTelegramBotToken',
            '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11',  # Example bot token from Telegram documentation
            '',
            None
        ]

        def is_valid_token(token):
            return token not in invalid_tokens and len(token.split(':')) == 2

        # Define retrieval methods
        def retrieve_from_env():
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            if bot_token and is_valid_token(bot_token):
                logging.info("Bot token successfully retrieved from environment variable.")
                return bot_token
            else:
                logging.warning("Invalid or unset TELEGRAM_BOT_TOKEN environment variable.")
                return None

        def retrieve_from_file():
            if token_file_path.is_file():
                try:
                    bot_token = token_file_path.read_text().strip()
                    if bot_token and is_valid_token(bot_token):
                        logging.info("Bot token successfully retrieved from bot_token.txt.")
                        return bot_token
                    else:
                        logging.error("Invalid or empty bot_token.txt.")
                        return None
                except IOError as e:
                    logging.error(f"Failed to read bot_token.txt. Details: {e}")
                    return None
            else:
                logging.error(f"bot_token.txt not found at {token_file_path}.")
                return None

        def query_user_for_token():
            logging.info("No valid bot token found. Please obtain a Telegram bot token from @BotFather on Telegram and paste it below.")
            logging.info("Press Enter without typing anything to quit.")
            token = input("Your Telegram bot token: ").strip()
            if token and is_valid_token(token):
                # Save the token to bot_token.txt for future use
                try:
                    token_file_path.write_text(token)
                    logging.info(f"Bot token saved to {token_file_path}.")
                    return token
                except IOError as e:
                    logging.error(f"Failed to save bot token to bot_token.txt. Details: {e}")
                    return None
            else:
                logging.error("No valid token entered. Exiting application.")
                sys.exit(1)

        # Retrieval logic based on configuration
        if prefer_env:
            token = retrieve_from_env()
            if token:
                return token
            elif allow_fallback:
                logging.warning("Preferred environment variable not found or invalid. Attempting to retrieve bot token from bot_token.txt as fallback.")
                token = retrieve_from_file()
                if token:
                    return token
                elif ask_for_token:
                    token = query_user_for_token()
                    if token:
                        return token
                    else:
                        raise BotTokenError("Failed to retrieve bot token from environment variable, token file, and user input.")
                else:
                    raise BotTokenError("Failed to retrieve bot token from both environment variable and token file.")
            else:
                logging.error("Environment variable not found and fallback is disabled.")
                raise BotTokenError(
                    "Failed to retrieve bot token. "
                    "Please ensure the TELEGRAM_BOT_TOKEN environment variable is set, or allow fallback by enabling it in config.ini."
                )
        else:
            token = retrieve_from_file()
            if token:
                return token
            elif allow_fallback:
                logging.warning("bot_token.txt not found or invalid. Attempting to retrieve bot token from environment variable as fallback.")
                token = retrieve_from_env()
                if token:
                    return token
                elif ask_for_token:
                    token = query_user_for_token()
                    if token:
                        return token
                    else:
                        raise BotTokenError("Failed to retrieve bot token from token file, environment variable, and user input.")
                else:
                    raise BotTokenError("Failed to retrieve bot token from both token file and environment variable.")
            else:
                logging.error("Token file not found and fallback is disabled.")
                raise BotTokenError(
                    "Failed to retrieve bot token. "
                    "Please ensure bot_token.txt exists at the expected location, or allow fallback by enabling it in config.ini."
                )

    except BotTokenError as e:
        logging.error(f"BotTokenError: {e}")
        sys.stderr.flush()  # Ensure all stderr logs are flushed
        sys.exit(1)  # Explicitly exit on BotTokenError
    except Exception as e:
        logging.error(f"Unexpected error while retrieving bot token: {e}")
        sys.stderr.flush()  # Ensure all stderr logs are flushed
        sys.exit(1)  # Explicitly exit on unexpected errors

# Example usage
if __name__ == "__main__":
    try:
        token = get_bot_token()
        logging.info("Bot token successfully retrieved.")
    except Exception as e:
        logging.critical("Failed to retrieve bot token. Exiting application.")
        sys.stderr.flush()  # Ensure all stderr logs are flushed
        sys.exit(1)

# # ((newer method; no query))
# # to be cleaned up, kept only for reference atm!
# # ~~~ Enhanced Read Telegram Bot Token with Configurable Fallback and Appropriate Logging ~~~

# import os
# import configparser
# import logging
# from pathlib import Path
# import sys

# # Set up basic logging configuration
# # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# class BotTokenError(Exception):
#     """Custom exception for bot token retrieval failures."""
#     pass

# def get_bot_token():
#     try:
#         # Ascend two levels to get the project root from bot_token.py in src/utils
#         base_dir = Path(__file__).resolve().parents[2]
#         config_path = base_dir / 'config' / 'config.ini'
#         token_file_path = base_dir / 'config' / 'bot_token.txt'

#         logging.debug(f"Base directory: {base_dir}")
#         logging.debug(f"Config path: {config_path}")
#         logging.debug(f"Token file path: {token_file_path}")

#         # Verify config.ini exists
#         if not config_path.is_file():
#             raise BotTokenError(f"config.ini not found at {config_path}.")

#         # Read configuration
#         config = configparser.ConfigParser()
#         config.read(config_path)

#         # Validate configuration
#         if 'DEFAULT' not in config:
#             raise BotTokenError("Missing 'DEFAULT' section in config.ini.")

#         prefer_env = config.getboolean('DEFAULT', 'PreferEnvForBotToken', fallback=True)
#         allow_fallback = config.getboolean('DEFAULT', 'AllowFallback', fallback=True)

#         # Define retrieval methods
#         def retrieve_from_env():
#             bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
#             if bot_token:
#                 logging.info("Bot token successfully retrieved from environment variable.")
#                 return bot_token
#             else:
#                 logging.warning("TELEGRAM_BOT_TOKEN environment variable not set.")
#                 return None

#         def retrieve_from_file():
#             if token_file_path.is_file():
#                 try:
#                     bot_token = token_file_path.read_text().strip()
#                     if bot_token:
#                         logging.info("Bot token successfully retrieved from bot_token.txt.")
#                         return bot_token
#                     else:
#                         logging.error("bot_token.txt is empty.")
#                         return None
#                 except IOError as e:
#                     logging.error(f"Failed to read bot_token.txt. Details: {e}")
#                     return None
#             else:
#                 logging.error(f"bot_token.txt not found at {token_file_path}.")
#                 return None

#         # Retrieval logic based on configuration
#         if prefer_env:
#             token = retrieve_from_env()
#             if token:
#                 return token
#             elif allow_fallback:
#                 logging.warning("Preferred environment variable not found. Attempting to retrieve bot token from bot_token.txt as fallback.")
#                 token = retrieve_from_file()
#                 if token:
#                     return token
#                 else:
#                     raise BotTokenError("Failed to retrieve bot token from both environment variable and token file.")
#             else:
#                 logging.error("Environment variable not found and fallback is disabled.")
#                 raise BotTokenError(
#                     "Failed to retrieve bot token. "
#                     "Please ensure the TELEGRAM_BOT_TOKEN environment variable is set, or allow fallback by enabling it in config.ini."
#                 )
#         else:
#             token = retrieve_from_file()
#             if token:
#                 return token
#             elif allow_fallback:
#                 logging.warning("bot_token.txt not found. Attempting to retrieve bot token from environment variable as fallback.")
#                 token = retrieve_from_env()
#                 if token:
#                     return token
#                 else:
#                     raise BotTokenError("Failed to retrieve bot token from both token file and environment variable.")
#             else:
#                 logging.error("Token file not found and fallback is disabled.")
#                 raise BotTokenError(
#                     "Failed to retrieve bot token. "
#                     "Please ensure bot_token.txt exists at the expected location, or allow fallback by enabling it in config.ini."
#                 )

#     except BotTokenError as e:
#         logging.error(f"BotTokenError: {e}")
#         sys.stderr.flush()  # Ensure all stderr logs are flushed
#         sys.exit(1)  # Explicitly exit on BotTokenError
#     except Exception as e:
#         logging.error(f"Unexpected error while retrieving bot token: {e}")
#         sys.stderr.flush()  # Ensure all stderr logs are flushed
#         sys.exit(1)  # Explicitly exit on unexpected errors

# # Example usage
# if __name__ == "__main__":
#     try:
#         token = get_bot_token()
#         logging.info("Bot token successfully retrieved.")
#     except Exception as e:
#         logging.critical("Failed to retrieve bot token. Exiting application.")
#         sys.stderr.flush()  # Ensure all stderr logs are flushed
#         sys.exit(1)

# # # // (old method)
# # # ~~~ read the telegram bot token ~~~

# # import os
# # import configparser
# # import sys
# # import logging

# # def get_bot_token():
# #     # Correctly ascend two levels to get the project root from bot_token.py in src/utils
# #     base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# #     config_path = os.path.join(base_dir, 'config', 'config.ini')
# #     token_file_path = os.path.join(base_dir, 'config', 'bot_token.txt')

# #     logging.info(f"Debug: Base directory is {base_dir}")
# #     logging.info(f"Debug: Config path is {config_path}")
# #     logging.info(f"Debug: Token file path is {token_file_path}")

# #     # Check if the paths actually exist
# #     if not os.path.exists(config_path):
# #         print("Error: config.ini not found at the expected path.")
# #         sys.exit(1)

# #     if not os.path.exists(token_file_path) and not os.getenv('TELEGRAM_BOT_TOKEN'):
# #         print("Error: bot_token.txt not found at the expected path and TELEGRAM_BOT_TOKEN environment variable is not set.")
# #         sys.exit(1)

# #     # Reading the config
# #     config = configparser.ConfigParser()
# #     config.read(config_path)
# #     prefer_env = config.getboolean('DEFAULT', 'PreferEnvForBotToken', fallback=True)

# #     if prefer_env and os.getenv('TELEGRAM_BOT_TOKEN'):
# #         return os.getenv('TELEGRAM_BOT_TOKEN')

# #     # Try to read the token from the file if the environment variable isn't preferred or set
# #     try:
# #         with open(token_file_path, 'r') as file:
# #             return file.read().strip()
# #     except FileNotFoundError:
# #         print("Error: Failed to read bot_token.txt.")
# #         sys.exit(1)

# #     # Fallback error message
# #     print("The bot token could not be determined.")
# #     sys.exit(1)

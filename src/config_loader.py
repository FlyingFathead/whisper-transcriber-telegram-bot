# config_loader.py 
# (update to include RateLimitSettings and NotificationSettings)

import configparser
import os
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._config = configparser.ConfigParser()
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'config.ini')
            
            if os.path.exists(config_path):
                cls._config.read(config_path)
                logger.info(f"Loaded config from {config_path}")
                for section in cls._config.sections():
                    logger.debug(f"Section [{section}]: {dict(cls._config[section])}")
            else:
                logger.warning(f"Config file not found at {config_path}")
        return cls._instance

    @classmethod
    def get_config(cls):
        if cls._config is None:
            cls()
        return cls._config

    # NEW: Method to get Notification Settings
    @classmethod
    def get_notification_settings(cls):
        config = cls.get_config()  # Get the config object
        send_completion_message = config.getboolean('NotificationSettings', 'sendcompletionmessage', fallback=True)
        completion_message = config.get('NotificationSettings', 'completionmessage', fallback="Transcription complete. Have a nice day!")
        
        return {
            'send_completion_message': send_completion_message,
            'completion_message': completion_message
        }

    # NEW: Method to get yt-dlp domain settings
    @classmethod
    def get_ytdlp_domain_settings(cls):
        config = cls.get_config()
        active = config.getboolean('YTDLPSettings', 'download_original_video_for_domains_active', fallback=False)
        domains = config.get('YTDLPSettings', 'download_original_video_domains', fallback='')
        # Split by comma and strip whitespace
        domain_list = [domain.strip().lower() for domain in domains.split(',') if domain.strip()]
        
        return {
            'active': active,
            'domains': domain_list
        }

# Usage example:
# from config_loader import ConfigLoader
# notification_settings = ConfigLoader.get_notification_settings()

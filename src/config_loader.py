# config_loader.py

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

# Usage:
# from config_loader import ConfigLoader
# config = ConfigLoader.get_config()

# config_loader.py 
# (update to include RateLimitSettings and NotificationSettings)

import configparser
import os
import logging

logger = logging.getLogger(__name__)

# newline parser when needed
def _parse_newlines(s: str) -> str:
    """
    Replace literal backslash-n sequences (\\n) with actual newlines.
    """
    return s.replace("\\n", "\n")

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

    @classmethod
    def get_transcription_settings(cls):
        config = cls.get_config()
        transcription_settings = {
            'include_header': config.getboolean('TranscriptionSettings', 'includeheaderintranscription', fallback=False),
            'keep_audio_files': config.getboolean('TranscriptionSettings', 'keepaudiofiles', fallback=False),
            'send_as_files': config.getboolean('TranscriptionSettings', 'sendasfiles', fallback=True),
            'send_timestamped_txt': config.getboolean('TranscriptionSettings', 'send_timestamped_txt', fallback=False), # ADDED, default to False
            'shorten_timestamps_under_one_hour': config.getboolean('TranscriptionSettings', 'shorten_timestamps_under_one_hour', fallback=True), # ADDED THIS LINE in v0.1717            
            'send_as_messages': config.getboolean('TranscriptionSettings', 'sendasmessages', fallback=False),
        }
        logger.info(f"Loaded transcription settings: {transcription_settings}")
        return transcription_settings

    # NEW: Method to get Notification Settings
    @classmethod
    def get_notification_settings(cls):
        config = cls.get_config()  # Get the config object
        send_completion_message = config.getboolean('NotificationSettings', 'sendcompletionmessage', fallback=True)
        # completion_message = config.get('NotificationSettings', 'completionmessage', fallback="Transcription complete. Have a nice day!")

        # Read the raw string from config
        completion_message_raw = config.get('NotificationSettings', 'completionmessage', fallback="Transcription complete.")

        # Convert \n → actual newlines
        completion_message = _parse_newlines(completion_message_raw)
        
        # === New lines for your queue, GPU, and audio info messages ===
        queue_message_next = config.get(
            'NotificationSettings', 'queue_message_next',
            fallback="⏳ Your request is next and is currently being processed."
        )
        queue_message_queued = config.get(
            'NotificationSettings', 'queue_message_queued',
            fallback="⏳ Your request has been added to the queue. There are {jobs_ahead} jobs ahead of yours."
        )
        audio_info_message_raw = config.get(
            'NotificationSettings', 'audio_info_message',
            fallback="Audio file length:\n{audio_duration}\n\nWhisper model in use:\n{model}\n\n"
                     "Model language set to:\n{language}\n\nEstimated transcription time:\n{est_time:.1f} minutes.\n\n"
                     "Transcribing audio..."
        )

        # Convert \n → actual newlines
        audio_info_message = _parse_newlines(audio_info_message_raw)

        gpu_message_template_raw = config.get(
            'NotificationSettings', 'gpu_message_template',
            fallback="Using GPU {gpu_id}: {gpu_name}\nFree Memory: {gpu_free} MB\nLoad: {gpu_load}%"
        )

        # Convert \n → actual newlines
        gpu_message_template = _parse_newlines(gpu_message_template_raw)

        gpu_message_no_gpu = config.get(
            'NotificationSettings', 'gpu_message_no_gpu',
            fallback="⚠️ No GPU available, using CPU for transcription. WARNING: this will be slow."
        )

        send_video_info = config.getboolean('NotificationSettings', 'send_video_info', fallback=True)
        send_detailed_info = config.getboolean('NotificationSettings', 'send_detailed_info', fallback=True)

        # voice msg and audio file handling
        voice_message_received = config.get('NotificationSettings', 'voice_message_received', fallback="")
        audio_file_received    = config.get('NotificationSettings', 'audio_file_received',    fallback="")

        return {
            'queue_message_next': queue_message_next,
            'queue_message_queued': queue_message_queued,
            'audio_info_message': audio_info_message,
            'gpu_message_template': gpu_message_template,
            'gpu_message_no_gpu': gpu_message_no_gpu,
            'send_video_info': send_video_info,
            'send_detailed_info': send_detailed_info,
            'voice_message_received': voice_message_received,
            'audio_file_received': audio_file_received,            
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

    @classmethod
    def get_special_domain_commands(cls):
        """
        Returns a dict of domain -> custom yt-dlp argument string,
        parsed from 'special_domain_commands' in the [YTDLPSettings] section.
        """
        config = cls.get_config()

        # Only parse them if usage is enabled
        enabled = config.getboolean("YTDLPSettings", "use_special_commands_for_domains", fallback=False)
        if not enabled:
            return {}  # No special commands if disabled

        raw = config.get("YTDLPSettings", "special_domain_commands", fallback="").strip()
        if not raw:
            return {}

        commands = {}
        for line in raw.splitlines():
            line = line.strip()
            # Skip empty lines or comment lines if you want
            if not line or line.startswith("#"):
                continue
            if '|' not in line:
                continue
            domain, args = line.split('|', 1)
            domain = domain.strip().lower()
            args = args.strip()
            commands[domain] = args
        return commands

    # get the owner ID's and ping on startup if needed
    @classmethod
    def get_owner_ids(cls):
        """ Return a list of owner IDs as integers. """
        config = cls.get_config()
        use_env = config.getboolean('OwnerSettings', 'use_env_for_ownerid', fallback=False)
        env_var_name = config.get('OwnerSettings', 'ownerid_env_var_name', fallback='WHISPER_TRANSCRIBERBOT_OWNER_USERID')
        fallback_str = config.get('OwnerSettings', 'ownerid_fallback', fallback='')

        if use_env:
            # Example: export WHISPER_TRANSCRIBERBOT_OWNER_USERID="12345,67890"
            env_val = os.getenv(env_var_name, '')
            if env_val.strip():
                logger.info(f"Reading owner ID(s) from env var {env_var_name}: {env_val}")
                try:
                    owners = [int(x.strip()) for x in env_val.split(',') if x.strip()]
                    return owners
                except ValueError:
                    logger.error(f"Could not parse environment variable {env_var_name} as int list: {env_val}")
                    return []
            else:
                logger.warning(f"Environment variable {env_var_name} is empty or not set; using fallback from config.ini")
        
        # Fallback branch: read from config
        if fallback_str.strip():
            logger.info(f"Using fallback owner IDs from config.ini: {fallback_str}")
            try:
                owners = [int(x.strip()) for x in fallback_str.split(',') if x.strip()]
                return owners
            except ValueError:
                logger.error(f"Could not parse fallback owner IDs in config.ini: {fallback_str}")
                return []
        
        logger.warning("No valid owner IDs found from environment or config.")
        return []

    @classmethod
    def should_ping_owners_on_start(cls):
        """ Return True/False if we should ping owners when the bot starts. """
        config = cls.get_config()
        return config.getboolean('OwnerSettings', 'ping_owners_on_start', fallback=False)

# Usage example:
# from config_loader import ConfigLoader
# notification_settings = ConfigLoader.get_notification_settings()

import aiohttp
import asyncio
import logging
import os
import ssl
import mimetypes
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class WhisperAPIError(Exception):
    pass

def get_audio_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('audio/'):
        return mime_type

    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.wma': 'audio/x-ms-wma',
        '.aiff': 'audio/aiff',
        '.webm': 'audio/webm',
    }
    return mime_map.get(ext, 'application/octet-stream')

async def transcribe_via_api(
    api_url: str,
    audio_path: str,
    model: str,
    language: str,
    output_format: str,
    api_engine: str = "faster_whisper",
    vad_filter: bool = False,
    word_timestamps: bool = False,
    diarize: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    timeout: int = 300,
    retry_attempts: int = 3,
    verify_ssl: bool = True
) -> str:
    if not os.path.exists(audio_path):
        raise WhisperAPIError(f"Audio file not found: {audio_path}")

    endpoint = f"{api_url.rstrip('/')}/asr"

    params = {
        "task": "transcribe",
        "output": output_format,
        "encode": "true"
    }

    if language and language != "auto":
        params["language"] = language

    if vad_filter and api_engine == "faster_whisper":
        params["vad_filter"] = "true"

    if word_timestamps and api_engine == "faster_whisper":
        params["word_timestamps"] = "true"

    if diarize and api_engine == "whisperx":
        params["diarize"] = "true"
        if min_speakers is not None:
            params["min_speakers"] = str(min_speakers)
        if max_speakers is not None:
            params["max_speakers"] = str(max_speakers)

    logger.info(f"API request params: {params}")

    ssl_context = None if verify_ssl else False
    mime_type = get_audio_mime_type(audio_path)
    filename = os.path.basename(audio_path)

    with open(audio_path, 'rb') as audio_file:
        file_content = audio_file.read()


    for attempt in range(retry_attempts):
        try:
            logger.info(f"Attempting API request (attempt {attempt + 1}/{retry_attempts}) to {endpoint}")

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                data = aiohttp.FormData()
                data.add_field('audio_file',
                             file_content,
                             filename=filename,
                             content_type=mime_type)

                async with session.post(
                    endpoint,
                    data=data,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"API request successful for format {output_format}")
                        return content
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        raise WhisperAPIError(f"API returned status {response.status}: {error_text}")
        except asyncio.TimeoutError:
            logger.error(f"API request timeout (attempt {attempt + 1}/{retry_attempts})")
            if attempt < retry_attempts - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                raise WhisperAPIError("API request timed out after all retry attempts")
        except aiohttp.ClientError as e:
            logger.error(f"API request failed (attempt {attempt + 1}/{retry_attempts}): {e}")
            if attempt < retry_attempts - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                raise WhisperAPIError(f"API request failed after all retry attempts: {e}")

        except Exception as e:
            logger.error(f"Unexpected error during API request: {e}")
            raise WhisperAPIError(f"Unexpected error: {e}")

    raise WhisperAPIError("Failed to transcribe via API after all attempts")

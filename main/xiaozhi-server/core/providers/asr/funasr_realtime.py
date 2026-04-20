import os
import asyncio
import threading
import traceback
import opuslib_next
import dashscope

from typing import TYPE_CHECKING
from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()


class _DashScopeRecognitionCallback(RecognitionCallback):
    def __init__(self, provider: "ASRProvider", conn: "ConnectionHandler"):
        super().__init__()
        self.provider = provider
        self.conn = conn

    def on_open(self) -> None:
        self.provider.server_ready = True
        logger.bind(tag=TAG).debug("FunASR-Realtime 连接已建立")

    def on_close(self) -> None:
        self.provider.server_ready = False
        logger.bind(tag=TAG).debug("FunASR-Realtime 连接已关闭")

    def on_complete(self) -> None:
        logger.bind(tag=TAG).debug("FunASR-Realtime 识别完成")
        self.provider._handle_complete(self.conn)

    def on_error(self, message) -> None:
        request_id = getattr(message, "request_id", "")
        error_message = getattr(message, "message", str(message))
        logger.bind(tag=TAG).error(
            f"FunASR-Realtime 出错: request_id={request_id}, message={error_message}"
        )
        self.provider._handle_error(self.conn)

    def on_event(self, result: RecognitionResult) -> None:
        self.provider._handle_event(self.conn, result)


class ASRProvider(ASRProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__()
        self.interface_type = InterfaceType.STREAM

        self.api_key = config.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("FunASR-Realtime 需要 api_key，或设置 DASHSCOPE_API_KEY")

        self.model = config.get("model", "fun-asr-realtime")
        self.format = config.get("format", "pcm")
        self.sample_rate = int(config.get("sample_rate", 16000))
        self.semantic_punctuation_enabled = bool(
            config.get("semantic_punctuation_enabled", False)
        )
        self.vocabulary_id = config.get("vocabulary_id")
        self.ws_url = config.get(
            "ws_url", "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
        )

        self.output_dir = config.get("output_dir", "tmp/")
        self.delete_audio_file = delete_audio_file

        self.decoder = opuslib_next.Decoder(16000, 1)
        self.recognition = None
        self.server_ready = False
        self.is_processing = False
        self._result_handled = False
        self._final_text_segments = []
        self._lock = threading.Lock()

    async def open_audio_channels(self, conn):
        await super().open_audio_channels(conn)

    async def receive_audio(self, conn, audio, audio_have_voice):
        await super().receive_audio(conn, audio, audio_have_voice)

        if audio_have_voice and not self.is_processing and self.recognition is None:
            await self._start_recognition(conn)

        if not self.is_processing or not self.server_ready or self.recognition is None:
            return

        try:
            if conn.audio_format == "pcm":
                pcm_frame = audio
            else:
                pcm_frame = self.decoder.decode(audio, 960)
            if pcm_frame:
                self.recognition.send_audio_frame(pcm_frame)
        except Exception as e:
            logger.bind(tag=TAG).warning(f"发送音频到 FunASR-Realtime 失败: {e}")
            await self._cleanup()

    async def _start_recognition(self, conn: "ConnectionHandler"):
        if self.recognition is not None or self.is_processing:
            return

        dashscope.api_key = self.api_key
        dashscope.base_websocket_api_url = self.ws_url

        callback = _DashScopeRecognitionCallback(self, conn)
        kwargs = {
            "model": self.model,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "semantic_punctuation_enabled": self.semantic_punctuation_enabled,
            "callback": callback,
        }
        if self.vocabulary_id:
            kwargs["vocabulary_id"] = self.vocabulary_id

        self.recognition = Recognition(**kwargs)
        self.is_processing = True
        self.server_ready = False
        self._result_handled = False
        self._final_text_segments = []
        logger.bind(tag=TAG).debug("启动 FunASR-Realtime 识别会话")
        await asyncio.to_thread(self.recognition.start)

    def _append_final_text(self, text: str):
        normalized = text.strip()
        if not normalized:
            return
        with self._lock:
            if not self._final_text_segments or self._final_text_segments[-1] != normalized:
                self._final_text_segments.append(normalized)

    def _current_text(self) -> str:
        with self._lock:
            return "".join(self._final_text_segments)

    def _trigger_handle_voice_stop(self, conn: "ConnectionHandler"):
        with self._lock:
            if self._result_handled:
                return
            self._result_handled = True

        audio_data = conn.asr_audio.copy()
        future = asyncio.run_coroutine_threadsafe(
            self.handle_voice_stop(conn, audio_data),
            conn.loop,
        )
        try:
            future.result()
        except Exception:
            logger.bind(tag=TAG).error(
                f"触发语音结束处理失败: {traceback.format_exc()}"
            )

    def _handle_event(self, conn: "ConnectionHandler", result: RecognitionResult):
        try:
            sentence = result.get_sentence()
            text = sentence.get("text", "").strip()
            if not text:
                return

            if RecognitionResult.is_sentence_end(sentence):
                logger.bind(tag=TAG).info(f"FunASR 最终识别文本: {text}")
                self._append_final_text(text)
                if conn.client_listen_mode == "manual":
                    if conn.client_voice_stop:
                        self._trigger_handle_voice_stop(conn)
                else:
                    self._trigger_handle_voice_stop(conn)
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理 FunASR 回调失败: {e}")

    def _handle_complete(self, conn: "ConnectionHandler"):
        if conn.client_listen_mode == "manual" and conn.client_voice_stop:
            self._trigger_handle_voice_stop(conn)

    def _handle_error(self, conn: "ConnectionHandler"):
        self.server_ready = False
        self.is_processing = False

    def stop_ws_connection(self):
        recognition = self.recognition
        self.recognition = None
        self.server_ready = False
        self.is_processing = False
        if recognition is not None:
            try:
                recognition.stop()
            except Exception as e:
                logger.bind(tag=TAG).debug(f"停止 FunASR-Realtime 连接失败: {e}")

    async def _send_stop_request(self):
        if self.recognition is None:
            return

        self.is_processing = False
        try:
            await asyncio.to_thread(self.recognition.stop)
        except Exception as e:
            logger.bind(tag=TAG).warning(f"停止 FunASR-Realtime 失败: {e}")

    async def speech_to_text(self, opus_data, session_id, audio_format, artifacts=None):
        text = self._current_text()
        self._final_text_segments = []
        return text, None

    async def _cleanup(self):
        recognition = self.recognition
        self.recognition = None
        self.server_ready = False
        self.is_processing = False
        if recognition is not None:
            try:
                await asyncio.to_thread(recognition.stop)
            except Exception:
                pass

    async def close(self):
        await self._cleanup()
        if self.decoder is not None:
            try:
                del self.decoder
            except Exception:
                pass

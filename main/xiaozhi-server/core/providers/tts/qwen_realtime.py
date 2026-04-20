import os
import queue
import base64
import asyncio
import traceback
import dashscope

from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType, ContentType, InterfaceType

from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)

TAG = __name__
logger = setup_logging()


class _QwenRealtimeCallback(QwenTtsRealtimeCallback):
    def __init__(self, provider: "TTSProvider"):
        super().__init__()
        self.provider = provider

    def on_open(self) -> None:
        logger.bind(tag=TAG).debug("Qwen-Realtime-TTS 连接已建立")

    def on_close(self, close_status_code, close_msg) -> None:
        logger.bind(tag=TAG).debug(
            f"Qwen-Realtime-TTS 连接关闭: code={close_status_code}, msg={close_msg}"
        )
        self.provider._active = False

    def on_event(self, response) -> None:
        self.provider._handle_event(response)


class TTSProvider(TTSProviderBase):
    TTS_PARAM_CONFIG = [
        ("ttsRate", "speed_ratio", 0.5, 2.0, 1.0, lambda v: round(float(v), 1)),
    ]

    AUDIO_FORMAT_MAP = {
        "pcm_24000": AudioFormat.PCM_24000HZ_MONO_16BIT,
    }

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.interface_type = InterfaceType.DUAL_STREAM
        self.report_on_last = True

        self.api_key = config.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Qwen-Realtime-TTS 需要 api_key，或设置 DASHSCOPE_API_KEY"
            )

        self.model = config.get("model", "qwen3-tts-instruct-flash-realtime")
        self.voice = config.get("voice", "Cherry")
        if config.get("private_voice"):
            self.voice = config.get("private_voice")

        self.instructions = config.get("instructions")
        self.optimize_instructions = bool(config.get("optimize_instructions", False))
        self.ws_url = config.get(
            "ws_url", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
        )
        self.response_format_name = config.get("response_format", "pcm_24000")
        self.response_format = self.AUDIO_FORMAT_MAP.get(
            self.response_format_name, AudioFormat.PCM_24000HZ_MONO_16BIT
        )
        self.audio_file_type = "pcm"
        self.speed_ratio = float(config.get("speed_ratio", 1.0))

        self.qwen_tts = None
        self._active = False
        self._first_packet_sent = False
        self._last_packet_sent = False
        self._current_text = ""
        self._audio_chunk_count = 0

        self._apply_percentage_params(config)

    def tts_text_priority_thread(self):
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)

                if self.conn.client_abort:
                    try:
                        logger.bind(tag=TAG).info("收到打断信息，终止 Qwen-Realtime-TTS")
                        asyncio.run_coroutine_threadsafe(
                            self.finish_session(self.conn.sentence_id),
                            loop=self.conn.loop,
                        )
                        continue
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"取消 Qwen-Realtime-TTS 失败: {e}")
                        continue

                if message.sentence_id != self.conn.sentence_id:
                    continue

                if message.sentence_type == SentenceType.FIRST:
                    self.current_sentence_id = message.sentence_id
                    self._current_text = ""
                    self._first_packet_sent = False
                    self._last_packet_sent = False
                    self._audio_chunk_count = 0
                    future = asyncio.run_coroutine_threadsafe(
                        self.start_session(message.sentence_id),
                        loop=self.conn.loop,
                    )
                    future.result(timeout=self.tts_timeout)

                elif message.content_type == ContentType.TEXT and message.content_detail:
                    self._current_text += message.content_detail
                    future = asyncio.run_coroutine_threadsafe(
                        self.text_to_speak(message.content_detail, None),
                        loop=self.conn.loop,
                    )
                    future.result(timeout=self.tts_timeout)

                elif message.content_type == ContentType.FILE:
                    logger.bind(tag=TAG).warning("Qwen-Realtime-TTS 不处理文件型 TTS 输入")

                if message.sentence_type == SentenceType.LAST:
                    future = asyncio.run_coroutine_threadsafe(
                        self.finish_session(message.sentence_id),
                        loop=self.conn.loop,
                    )
                    future.result(timeout=self.tts_timeout)

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理 Qwen-Realtime-TTS 文本失败: {e}, 详情: {traceback.format_exc()}"
                )
                continue

    def _handle_event(self, response) -> None:
        try:
            event_type = response.get("type")

            if event_type == "session.created":
                logger.bind(tag=TAG).debug("Qwen-Realtime-TTS session 已创建")
                return

            if event_type == "response.audio.delta":
                pcm_bytes = base64.b64decode(response["delta"])
                self._audio_chunk_count += 1
                if self._audio_chunk_count == 1:
                    logger.bind(tag=TAG).debug(
                        f"Qwen-Realtime-TTS 收到首个音频分片: {len(pcm_bytes)} bytes"
                    )
                if not self._first_packet_sent:
                    self.tts_audio_queue.put(
                        (
                            SentenceType.FIRST,
                            [],
                            self._current_text,
                            getattr(self, "current_sentence_id", None),
                        )
                    )
                    self._first_packet_sent = True
                self.opus_encoder.encode_pcm_to_opus_stream(
                    pcm_bytes,
                    end_of_stream=False,
                    callback=self.handle_opus,
                )
                return

            if event_type in ("response.done", "session.finished"):
                if self._last_packet_sent:
                    return
                self._active = False
                self._last_packet_sent = True
                logger.bind(tag=TAG).debug(
                    "Qwen-Realtime-TTS 响应结束: "
                    f"chunks={self._audio_chunk_count}, text_len={len(self._current_text)}"
                )
                self.tts_audio_queue.put(
                    (
                        SentenceType.LAST,
                        [],
                        self._current_text,
                        getattr(self, "current_sentence_id", None),
                    )
                )
                return
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理 Qwen-Realtime-TTS 事件失败: {e}")

    async def start_session(self, session_id):
        await self.close()

        dashscope.api_key = self.api_key
        self._active = True
        self._first_packet_sent = False
        self._last_packet_sent = False

        callback = _QwenRealtimeCallback(self)
        self.qwen_tts = QwenTtsRealtime(
            model=self.model,
            callback=callback,
            url=self.ws_url,
        )

        await asyncio.to_thread(self.qwen_tts.connect)
        session_params = {
            "voice": self.voice,
            "response_format": self.response_format,
            "mode": "server_commit",
        }
        if self.instructions:
            session_params["instructions"] = self.instructions
            session_params["optimize_instructions"] = self.optimize_instructions
        await asyncio.to_thread(self.qwen_tts.update_session, **session_params)
        logger.bind(tag=TAG).debug(f"Qwen-Realtime-TTS 会话已启动: {session_id}")

    async def text_to_speak(self, text, output_file):
        if not self.qwen_tts or not self._active:
            return
        logger.bind(tag=TAG).debug(f"Qwen-Realtime-TTS 追加文本: {text}")
        await asyncio.to_thread(self.qwen_tts.append_text, text)

    async def finish_session(self, session_id):
        if not self.qwen_tts:
            return
        try:
            await asyncio.to_thread(self.qwen_tts.finish)
        except Exception as e:
            logger.bind(tag=TAG).warning(f"结束 Qwen-Realtime-TTS 会话失败: {e}")

    async def close(self):
        if self.qwen_tts and self._active:
            try:
                await asyncio.to_thread(self.qwen_tts.finish)
            except Exception:
                pass
        await super().close()
        self._active = False
        self._first_packet_sent = False
        self._last_packet_sent = False
        self._current_text = ""
        self._audio_chunk_count = 0
        self.qwen_tts = None

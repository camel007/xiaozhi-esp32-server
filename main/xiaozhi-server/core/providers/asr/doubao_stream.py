import json
import gzip
import uuid
import asyncio
import websockets
import opuslib_next
from core.providers.asr.base import ASRProviderBase
from config.logger import setup_logging
from core.providers.asr.dto.dto import InterfaceType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__()
        self.interface_type = InterfaceType.STREAM
        self.config = config
        self.text = ""
        self.decoder = opuslib_next.Decoder(16000, 1)
        self.asr_ws = None
        self.forward_task = None
        self.is_processing = False  # 添加处理状态标志

        # 配置参数
        self.appid = str(config.get("appid"))
        self.cluster = config.get("cluster")
        self.access_token = config.get("access_token")
        self.boosting_table_name = config.get("boosting_table_name", "")
        self.correct_table_name = config.get("correct_table_name", "")
        self.output_dir = config.get("output_dir", "tmp/")
        self.delete_audio_file = delete_audio_file

        # 火山引擎ASR配置
        enable_multilingual = config.get("enable_multilingual", False)
        self.enable_multilingual = (
            False if str(enable_multilingual).lower() == "false" else True
        )
        configured_ws_url = config.get("ws_url")
        if configured_ws_url:
            self.ws_url = configured_ws_url
        elif self.enable_multilingual:
            self.ws_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
        else:
            self.ws_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"

        default_resource_id = (
            "volc.seedasr.sauc.duration"
            if "bigmodel_async" in self.ws_url
            else "volc.bigasr.sauc.duration"
        )
        self.resource_id = config.get("resource_id", default_resource_id)
        self.model_name = config.get(
            "model_name",
            "bigmodel2.0" if self.resource_id.startswith("volc.seedasr") else "bigmodel",
        )
        self.enable_nonstream = bool(config.get("enable_nonstream", True))
        self.enable_itn = bool(config.get("enable_itn", True))
        self.enable_punc = bool(config.get("enable_punc", True))
        self.enable_ddc = bool(config.get("enable_ddc", False))
        self.uid = config.get("uid", "streaming_asr_service")
        self.workflow = config.get(
            "workflow", "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate"
        )
        self.result_type = config.get("result_type", "single")
        self.format = config.get("format", "pcm")
        self.codec = config.get("codec", "pcm")
        self.rate = config.get("sample_rate", 16000)
        # language参数仅在多语种模式(bigmodel_nostream)下有效
        self.language = config.get("language") if self.enable_multilingual else None
        self.bits = config.get("bits", 16)
        self.channel = config.get("channel", 1)
        self.auth_method = config.get("auth_method", "token")
        self.secret = config.get("secret", "access_secret")
        end_window_size = config.get("end_window_size")
        self.end_window_size = int(end_window_size) if end_window_size else 200
        self.boosting_table_id = config.get("boosting_table_id", "")
        self.correct_table_id = config.get("correct_table_id", "")

    async def open_audio_channels(self, conn):
        await super().open_audio_channels(conn)

    async def receive_audio(self, conn: "ConnectionHandler", audio, audio_have_voice):
        # 先调用父类方法处理基础逻辑
        await super().receive_audio(conn, audio, audio_have_voice)
        
        # 如果本次有声音，且之前没有建立连接
        if audio_have_voice and self.asr_ws is None and not self.is_processing:
            try:
                self.is_processing = True
                # 建立新的WebSocket连接
                headers = self.token_auth() if self.auth_method == "token" else None
                logger.bind(tag=TAG).info(f"正在连接ASR服务，headers: {headers}")

                self.asr_ws = await websockets.connect(
                    self.ws_url,
                    additional_headers=headers,
                    max_size=1000000000,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=10,
                )

                # 发送初始化请求
                request_params = self.construct_request(str(uuid.uuid4()))
                try:
                    payload_bytes = str.encode(json.dumps(request_params))
                    payload_bytes = gzip.compress(payload_bytes)
                    full_client_request = self.generate_header()
                    full_client_request.extend((len(payload_bytes)).to_bytes(4, "big"))
                    full_client_request.extend(payload_bytes)

                    logger.bind(tag=TAG).info(f"发送初始化请求: {request_params}")
                    await self.asr_ws.send(full_client_request)

                    # 等待初始化响应
                    init_res = await self.asr_ws.recv()
                    result = self.parse_response(init_res)
                    logger.bind(tag=TAG).info(f"收到初始化响应: {result}")

                    # 检查初始化响应
                    if "code" in result and result["code"] != 1000:
                        error_msg = f"ASR服务初始化失败: {result.get('payload_msg', {}).get('error', '未知错误')}"
                        logger.bind(tag=TAG).error(error_msg)
                        raise Exception(error_msg)

                except Exception as e:
                    logger.bind(tag=TAG).error(f"发送初始化请求失败: {str(e)}")
                    if hasattr(e, "__cause__") and e.__cause__:
                        logger.bind(tag=TAG).error(f"错误原因: {str(e.__cause__)}")
                    raise e

                # 启动接收ASR结果的异步任务
                self.forward_task = asyncio.create_task(self._forward_asr_results(conn))

                # 发送缓存的音频数据
                if conn.asr_audio and len(conn.asr_audio) > 0:
                    for cached_audio in conn.asr_audio[-10:]:
                        try:
                            pcm_frame = self.decoder.decode(cached_audio, 960)
                            payload = gzip.compress(pcm_frame)
                            audio_request = bytearray(
                                self.generate_audio_default_header()
                            )
                            audio_request.extend(len(payload).to_bytes(4, "big"))
                            audio_request.extend(payload)
                            await self.asr_ws.send(audio_request)
                        except Exception as e:
                            logger.bind(tag=TAG).info(
                                f"发送缓存音频数据时发生错误: {e}"
                            )

            except Exception as e:
                logger.bind(tag=TAG).error(f"建立ASR连接失败: {str(e)}")
                if hasattr(e, "__cause__") and e.__cause__:
                    logger.bind(tag=TAG).error(f"错误原因: {str(e.__cause__)}")
                if self.asr_ws:
                    await self.asr_ws.close()
                    self.asr_ws = None
                self.is_processing = False
                return

        # 发送当前音频数据
        if self.asr_ws and self.is_processing:
            try:
                pcm_frame = self.decoder.decode(audio, 960)
                payload = gzip.compress(pcm_frame)
                audio_request = bytearray(self.generate_audio_default_header())
                audio_request.extend(len(payload).to_bytes(4, "big"))
                audio_request.extend(payload)
                await self.asr_ws.send(audio_request)
            except Exception as e:
                logger.bind(tag=TAG).info(f"发送音频数据时发生错误: {e}")

    async def _forward_asr_results(self, conn: "ConnectionHandler"):
        try:
            while self.asr_ws and not conn.stop_event.is_set():
                # 获取当前连接的音频数据
                audio_data = conn.asr_audio
                try:
                    response = await self.asr_ws.recv()
                    result = self.parse_response(response)
                    logger.bind(tag=TAG).debug(f"收到ASR结果: {result}")

                    if "payload_msg" in result:
                        payload = result["payload_msg"]
                        # 检查是否是错误码1013（无有效语音）
                        if "code" in payload and payload["code"] == 1013:
                            # 静默处理，不记录错误日志
                            continue

                        if "result" in payload:
                            utterances = payload["result"].get("utterances", [])
                            # 检查duration和空文本的情况
                            if (
                                not self.enable_multilingual  # 注意：多语种模式不返回中间结果，需要等待最终结果
                                and payload.get("audio_info", {}).get("duration", 0)
                                > 2000
                                and not utterances
                                and not payload["result"].get("text")
                                and conn.client_listen_mode != "manual"
                            ):
                                logger.bind(tag=TAG).error(f"识别文本：空")
                                self.text = ""
                                if len(audio_data) > 15:  # 确保有足够音频数据
                                    await self.handle_voice_stop(conn, audio_data)
                                break

                            # 专门处理没有文本的识别结果（手动模式下可能已经识别完成但是没松按键）
                            elif not payload["result"].get("text") and not utterances:
                                # 多语种模式会持续返回空文本，直到最后返回完整结果，所以需要排除
                                if self.enable_multilingual:
                                    continue

                                if conn.client_listen_mode == "manual" and conn.client_voice_stop and len(audio_data) > 15:
                                    logger.bind(tag=TAG).debug("消息结束收到停止信号，触发处理")
                                    await self.handle_voice_stop(conn, audio_data)
                                    break

                            for utterance in utterances:
                                if utterance.get("definite", False):
                                    current_text = utterance["text"]
                                    logger.bind(tag=TAG).info(
                                        f"识别到文本: {current_text}"
                                    )

                                    # 手动模式下累积识别结果
                                    if conn.client_listen_mode == "manual":
                                        if self.text:
                                            self.text += current_text
                                        else:
                                            self.text = current_text

                                        # 在接收消息中途时收到停止信号
                                        if conn.client_voice_stop and len(audio_data) > 0:
                                            logger.bind(tag=TAG).debug("消息中途收到停止信号，触发处理")
                                            await self.handle_voice_stop(conn, audio_data)
                                        break
                                    else:
                                        # 自动模式下直接覆盖
                                        self.text = current_text
                                        if len(audio_data) > 15:  # 确保有足够音频数据
                                            await self.handle_voice_stop(
                                                conn, audio_data
                                            )
                                    break
                        elif "error" in payload:
                            error_msg = payload.get("error", "未知错误")
                            logger.bind(tag=TAG).error(f"ASR服务返回错误: {error_msg}")
                            break

                except websockets.ConnectionClosed:
                    logger.bind(tag=TAG).info("ASR服务连接已关闭")
                    self.is_processing = False
                    break
                except Exception as e:
                    logger.bind(tag=TAG).error(f"处理ASR结果时发生错误: {str(e)}")
                    if hasattr(e, "__cause__") and e.__cause__:
                        logger.bind(tag=TAG).error(f"错误原因: {str(e.__cause__)}")
                    self.is_processing = False
                    break

        except Exception as e:
            logger.bind(tag=TAG).error(f"ASR结果转发任务发生错误: {str(e)}")
            if hasattr(e, "__cause__") and e.__cause__:
                logger.bind(tag=TAG).error(f"错误原因: {str(e.__cause__)}")
        finally:
            if self.asr_ws:
                await self.asr_ws.close()
                self.asr_ws = None
            self.is_processing = False
            # 重置所有音频相关状态
            conn.reset_audio_states()

    def stop_ws_connection(self):
        if self.asr_ws:
            asyncio.create_task(self.asr_ws.close())
            self.asr_ws = None
        self.is_processing = False

    async def _send_stop_request(self):
        """发送最后一个音频帧以通知服务器结束"""
        if self.asr_ws:
            try:
                # 发送结束标记的音频帧（gzip压缩的空数据）
                empty_payload = gzip.compress(b"")
                last_audio_request = bytearray(
                    self.generate_last_audio_default_header()
                )
                last_audio_request.extend(len(empty_payload).to_bytes(4, "big"))
                last_audio_request.extend(empty_payload)
                await self.asr_ws.send(last_audio_request)
                logger.bind(tag=TAG).debug("已发送结束音频帧")
            except Exception as e:
                logger.bind(tag=TAG).debug(f"发送结束音频帧时出错: {e}")

    def construct_request(self, reqid):
        request_config = {
            "reqid": reqid,
            "show_utterances": True,
            "result_type": self.result_type,
            "sequence": 1,
            "end_window_size": self.end_window_size,
        }

        # 双向流式优化版新增参数
        if "bigmodel_async" in self.ws_url:
            request_config.update(
                {
                    "model_name": self.model_name,
                    "enable_nonstream": self.enable_nonstream,
                    "enable_itn": self.enable_itn,
                    "enable_punc": self.enable_punc,
                    "enable_ddc": self.enable_ddc,
                }
            )
        else:
            request_config["workflow"] = self.workflow

        corpus = {}
        if self.boosting_table_id:
            corpus["boosting_table_id"] = self.boosting_table_id
        elif self.boosting_table_name:
            corpus["boosting_table_name"] = self.boosting_table_name
        if self.correct_table_id:
            corpus["correct_table_id"] = self.correct_table_id
        elif self.correct_table_name:
            corpus["correct_table_name"] = self.correct_table_name
        if corpus:
            request_config["corpus"] = corpus

        req = {
            "app": {
                "appid": self.appid,
                "cluster": self.cluster,
                "token": self.access_token,
            },
            "user": {"uid": self.uid},
            "request": request_config,
            "audio": {
                "format": self.format,
                "codec": self.codec,
                "rate": self.rate,
                "bits": self.bits,
                "channel": self.channel,
                "sample_rate": self.rate,
            },
        }

        # language参数仅在多语种模式下添加
        if self.enable_multilingual and self.language:
            req["audio"]["language"] = self.language

        logger.bind(tag=TAG).debug(
            f"构造请求参数: {json.dumps(req, ensure_ascii=False)}"
        )
        return req

    def token_auth(self):
        return {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    def generate_header(
        self,
        version=0x01,
        message_type=0x01,
        message_type_specific_flags=0x00,
        serial_method=0x01,
        compression_type=0x01,
        reserved_data=0x00,
        extension_header: bytes = b"",
    ):
        header = bytearray()
        header_size = int(len(extension_header) / 4) + 1
        header.append((version << 4) | header_size)
        header.append((message_type << 4) | message_type_specific_flags)
        header.append((serial_method << 4) | compression_type)
        header.append(reserved_data)
        header.extend(extension_header)
        return header

    def generate_audio_default_header(self):
        return self.generate_header(
            version=0x01,
            message_type=0x02,
            message_type_specific_flags=0x00,
            serial_method=0x01,
            compression_type=0x01,
        )

    def generate_last_audio_default_header(self):
        return self.generate_header(
            version=0x01,
            message_type=0x02,
            message_type_specific_flags=0x02,
            serial_method=0x01,
            compression_type=0x01,
        )

    def parse_response(self, res: bytes) -> dict:
        try:
            # 检查响应长度
            if len(res) < 4:
                logger.bind(tag=TAG).error(f"响应数据长度不足: {len(res)}")
                return {"error": "响应数据长度不足"}

            # 协议头解析
            header_size = (res[0] & 0x0F) * 4
            if len(res) < header_size:
                logger.bind(tag=TAG).error(
                    f"响应头长度异常: total={len(res)}, header_size={header_size}"
                )
                return {"error": "响应头长度异常"}
            message_type = (res[1] >> 4) & 0x0F
            compression_type = res[2] & 0x0F

            # 如果是错误响应
            if message_type == 0x0F:  # SERVER_ERROR_RESPONSE
                base = header_size
                if len(res) < base + 8:
                    return {"error": "错误响应长度不足"}
                code = int.from_bytes(res[base : base + 4], "big", signed=False)
                msg_length = int.from_bytes(
                    res[base + 4 : base + 8], "big", signed=False
                )
                payload = res[base + 8 : base + 8 + msg_length]
                if not payload:
                    payload = res[base + 8 :]
                if compression_type == 0x01 and payload.startswith(b"\x1f\x8b"):
                    payload = gzip.decompress(payload)
                error_msg = json.loads(payload.decode("utf-8"))
                return {
                    "code": code,
                    "msg_length": msg_length,
                    "payload_msg": error_msg,
                }

            # 兼容不同帧结构：
            # 1) [header][payload_size][payload]
            # 2) [header][seq][payload_size][payload]
            # 3) 历史兼容 [12字节后直接json]
            candidates = []

            if len(res) >= header_size + 4:
                payload_size = int.from_bytes(
                    res[header_size : header_size + 4], "big", signed=False
                )
                start = header_size + 4
                if len(res) >= start + payload_size:
                    candidates.append((start, payload_size))

            if len(res) >= header_size + 8:
                payload_size = int.from_bytes(
                    res[header_size + 4 : header_size + 8], "big", signed=False
                )
                start = header_size + 8
                if len(res) >= start + payload_size:
                    candidates.append((start, payload_size))

            if len(res) > 12:
                candidates.append((12, len(res) - 12))

            if len(res) > header_size:
                candidates.append((header_size, len(res) - header_size))

            tried = set()
            for start, size in candidates:
                key = (start, size)
                if key in tried:
                    continue
                tried.add(key)
                payload = res[start : start + size]
                if not payload:
                    continue
                try:
                    if compression_type == 0x01 and payload.startswith(b"\x1f\x8b"):
                        payload = gzip.decompress(payload)
                    json_data = payload.decode("utf-8")
                    result = json.loads(json_data)
                    logger.bind(tag=TAG).debug(f"成功解析JSON响应: {result}")
                    return {"payload_msg": result}
                except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                    continue

            logger.bind(tag=TAG).error(
                f"JSON解析失败: message_type={message_type}, header_size={header_size}, total={len(res)}"
            )
            logger.bind(tag=TAG).error(f"原始数据: {res}")
            raise ValueError("无法从响应中解析JSON")

        except Exception as e:
            logger.bind(tag=TAG).error(f"解析响应失败: {str(e)}")
            logger.bind(tag=TAG).error(f"原始响应数据: {res.hex()}")
            raise

    async def speech_to_text(self, opus_data, session_id, audio_format, artifacts=None):
        result = self.text
        self.text = ""  # 清空text
        return result, None

    async def close(self):
        """资源清理方法"""
        if self.asr_ws:
            await self.asr_ws.close()
            self.asr_ws = None
        if self.forward_task:
            self.forward_task.cancel()
            try:
                await self.forward_task
            except asyncio.CancelledError:
                pass
            self.forward_task = None
        self.is_processing = False

        # 显式释放decoder资源
        if hasattr(self, "decoder") and self.decoder is not None:
            try:
                del self.decoder
                self.decoder = None
                logger.bind(tag=TAG).debug("Doubao decoder resources released")
            except Exception as e:
                logger.bind(tag=TAG).debug(f"释放Doubao decoder资源时出错: {e}")

from typing import Dict, List

import torch

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # pragma: no cover
    process_vision_info = None
    _QWEN_VL_UTILS_IMPORT_ERROR = exc
else:
    _QWEN_VL_UTILS_IMPORT_ERROR = None


class LipReadingDataCollator:
    def __init__(self, processor, max_length: int = 2048, fps: float = 1.0, max_frames: int = 32):
        self.processor = processor
        self.max_length = max_length
        self.fps = fps
        self.max_frames = max_frames

    def _build_messages(self, sample: Dict[str, str]) -> List[Dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample["video_path"], "fps": self.fps, "max_frames": self.max_frames},
                    {"type": "text", "text": sample["instruction"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        if process_vision_info is None:
            raise ImportError(
                "qwen-vl-utils is required for video input processing."
                f" Original error: {_QWEN_VL_UTILS_IMPORT_ERROR}"
            )

        messages_list = [self._build_messages(sample) for sample in batch]

        full_texts: List[str] = []
        prompt_texts: List[str] = []
        videos = []
        video_metadatas = []
        video_kwargs = {}

        for messages in messages_list:
            full_texts.append(
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
            prompt_messages = messages[:1]
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            )

            _, video_inputs, current_video_kwargs = process_vision_info(
                [messages],
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if current_video_kwargs:
                video_kwargs = current_video_kwargs

            if video_inputs is None or len(video_inputs) != 1:
                raise ValueError("Each sample should contain exactly one video input.")
            video_item = video_inputs[0]
            if isinstance(video_item, tuple) and len(video_item) == 2:
                video_tensor, video_metadata = video_item
            else:
                video_tensor, video_metadata = video_item, None
            videos.append(video_tensor)
            video_metadatas.append(video_metadata)


        model_inputs = self.processor(
            text=full_texts,
            videos=videos,
            video_metadata=video_metadatas,
            **video_kwargs,
            padding=True,
            truncation=True,
            # max_length=self.max_length,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        for i, (prompt_text, video_input, video_metadata) in enumerate(
            zip(prompt_texts, videos, video_metadatas)
        ):
            prompt_inputs = self.processor(
                text=[prompt_text],
                videos=[video_input],
                video_metadata=[video_metadata],
                **video_kwargs,
                padding=False,
                truncation=True,
                # max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            labels[i, :prompt_len] = -100

        model_inputs["labels"] = labels
        return model_inputs

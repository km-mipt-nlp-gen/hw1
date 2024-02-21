from datetime import datetime
import torch


class ChatServiceUtil:
    @staticmethod
    def mean_pool(token_embeds: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
        return pool

    @staticmethod
    def timestamp_log(message):
        if not message:
            message = ""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")
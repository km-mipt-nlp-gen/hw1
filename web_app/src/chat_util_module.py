import sys
import logging
import torch


class ChatUtil:
    def __init__(self, logging_lvl, constants):
        logging.basicConfig(level=logging_lvl,
                            format=constants.LOG_FORMAT,
                            datefmt=constants.DATE_FORMAT,
                            handlers=[logging.StreamHandler(sys.stdout)])
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def mean_pool(token_embeds: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
        return pool

    def debug(self, message):
        if not message:
            message = ""
        self.logger.debug(message)

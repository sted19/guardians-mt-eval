r"""
InfoXLM Encoder
==============
    Pretrained InfoXLM encoder from Hugging Face.
"""
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizerFast

logger = logging.getLogger(__name__)


class InfoXLMEncoder(nn.Module):
    """InfoXLM encoder.

    InfoXLM shares the same architecture as XLM-RoBERTa but is pre-trained with
    additional cross-lingual contrastive and information-theoretic objectives.

    Args:
        pretrained_model (str): Pretrained model from hugging face (e.g. "microsoft/infoxlm-large").
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face.
        target_languages (Optional[List[str]]): List of target language special tokens to add.
    """

    def __init__(
        self,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        target_languages: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(pretrained_model)
        if load_pretrained_weights:
            self.model = XLMRobertaModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
        else:
            self.model = XLMRobertaModel(
                XLMRobertaConfig.from_pretrained(pretrained_model),
                add_pooling_layer=False,
            )
        self.model.encoder.output_hidden_states = False

        # If target_languages is provided, add them as special tokens.
        if target_languages:
            special_tokens_dict = {"additional_special_tokens": target_languages}
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added} target language special tokens.")

    @property
    def output_units(self) -> int:
        """Hidden size of the encoder model."""
        return self.model.config.hidden_size

    def prepare_sample(self, sample: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Receives a list of dictionaries (each with keys 'src' and, optionally, 'lp') and applies tokenization.
        It prepends a target language special token (e.g., "<target:en>") to the source text,
        based on the target language extracted from the "lp" field (if present).

        Args:
            sample (List[Dict[str, str]]): List of dictionaries, each containing a sample to tokenize.

        Returns:
            Dict[str, torch.Tensor]: Tokenized output (input_ids, attention_mask, etc.).
        """
        processed_texts = []
        for item in sample:
            text = item["src"]
            if "lp" in item:
                parts = item["lp"].split("-")
                assert len(parts) == 2
                target_lang = parts[1]
                # Construct the special token for the target language.
                target_token = f"<target:{target_lang}>"
                # Security check: ensure that target_token was added as a special token.
                if target_token not in self.tokenizer.all_special_tokens:
                    raise ValueError(
                        f"Target token {target_token} was not added as a special token in the tokenizer."
                    )
                # Prepend the target token to the source text.
                text = f"{target_token} {text}"
            processed_texts.append(text)

        print("Warning: Truncation is set explicitly to 512 tokens, since max_position_embeddings of InfoXLM is 514, leading to index out of bounds errors.")

        tokenizer_output = self.tokenizer(
            processed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return tokenizer_output

    @classmethod
    def from_pretrained(
        cls, pretrained_model: str, load_pretrained_weights: bool = True, target_languages: Optional[List[str]] = None,
    ) -> "InfoXLMEncoder":
        """Function that loads a pretrained InfoXLM encoder from Hugging Face.

        Args:
            pretrained_model (str): Name of the pretrained model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face.
            target_languages (Optional[List[str]]): List of target language special tokens to add.

        Returns:
            InfoXLMEncoder: InfoXLMEncoder object.
        """
        return InfoXLMEncoder(pretrained_model, load_pretrained_weights, target_languages)

    def freeze_embeddings(self) -> None:
        """Freezes the embedding layer."""
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_last_hidden_states=False,
    ) -> torch.Tensor:
        """Forward pass of the encoder.

        Args:
            input_ids (torch.Tensor): Tensor of input ids.
            attention_mask (torch.Tensor): Tensor of attention masks.
            return_last_hidden_states (bool): Flag to return all last hidden states. Default is False.
        Returns:
            torch.Tensor: Last hidden states tensor. If return_last_hidden_states is True, returns all last hidden
                          states. Otherwise, returns the last hidden state corresponding to the `[CLS]` token.
        """
        last_hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

        # If return_last_hidden_states is True, return all hidden states
        if return_last_hidden_states:
            return last_hidden_states
        else:
            # Return only the last hidden state corresponding to the `[CLS]` token
            return last_hidden_states[:, 0, :]

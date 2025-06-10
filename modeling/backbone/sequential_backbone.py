import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import fbgemm_gpu
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseSR
from .hstu_base_module import (DotProductSimilarity, RelativeBucketedTimeAndPositionBasedBias, SequentialTransductionUnitJagged, 
                               HSTUJagged, LocalNegativesSampler, HSTUCacheState, truncated_normal, L2NormEmbeddingPostprocessor, 
                               LearnablePositionalEmbeddingInputFeaturesPreprocessor)


class HSTU(BaseSR):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.model_name = "hstu"
        self.train_seq = dataset.seq_mat.cuda()
        self.embedding_dim: int = args.embedding_dim
        self._max_sequence_length: int = self.dataset.max_sequence_len
        self._num_blocks: int = args.num_blocks
        self._num_heads: int = args.num_heads
        self._dqk: int = args.attention_dim
        self._dv: int = args.linear_dim
        self._linear_activation: str = args.linear_activation
        self._linear_dropout_rate: float = args.linear_dropout_rate
        self._attn_dropout_rate: float = args.attn_dropout_rate
        self._enable_relative_attention_bias: bool = args.enable_relative_attention_bias
        self._concat_ua: bool = args.concat_ua
        self._normalization: str = args.normalization
        self._max_output_len: int = args.max_output_len
        self._linear_config = "uvqk"
        self._num_to_sample = args.num_negatives
        self.dropout_rate = args.dropout_rate
        self._softmax_temperature = args.temperature

        self.negatives_sampler = LocalNegativesSampler(all_item_ids=self.dataset.all_item_ids)
        self._ndp_module = DotProductSimilarity()
        self.item_emb = nn.Embedding(self.num_items + 1, self.embedding_dim, padding_idx=0)
        self._input_features_preproc = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            max_sequence_len=self._max_sequence_length + self._max_output_len,
            embedding_dim=self.embedding_dim,
            dropout_rate=self.dropout_rate,
        )
        self._output_postproc = L2NormEmbeddingPostprocessor(
            embedding_dim=self.embedding_dim,
            eps=1e-6,
        )
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self.embedding_dim,
                    linear_hidden_dim=self._dv,
                    attention_dim=self._dqk,
                    normalization=self._normalization,
                    linear_config=self._linear_config,
                    linear_activation=self._linear_activation,
                    num_heads=self._num_heads,
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=self._max_sequence_length
                            + self._max_output_len,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if self._enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=self._linear_dropout_rate,
                    attn_dropout_ratio=self._attn_dropout_rate,
                    concat_ua=self._concat_ua,
                )
                for _ in range(self._num_blocks)
            ],
            autocast_dtype=None,
        )
        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + self._max_output_len,
                        self._max_sequence_length + self._max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self.reset_para()
    
    def reset_para(self):
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params, mean=0.0, std=0.02)
            
            if ("_hstu" in name):
                continue

            try:
                torch.nn.init.xavier_normal_(params.data)
            except:
                print(f"Failed to initialize {name}: {params.data.size()} params")
    
    def _l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.clamp(
            torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
            min=1e-6,
        )
        return x
    
    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        [B, N] -> [B, N, D].
        """
        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()

        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        float_dtype = user_embeddings.dtype
        user_embeddings, cached_states = self._hstu(
            x=user_embeddings,
            x_offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths),
            all_timestamps=None,
            invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        return self._output_postproc(user_embeddings), cached_states
    
    def similarity_fn(
        self,
        query_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        item_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch._assert(
            len(query_embeddings.size()) == 2, "len(query_embeddings.size()) must be 2"
        )
        torch._assert(len(item_ids.size()) == 2, "len(item_ids.size()) must be 2")
        if item_embeddings is None:
            item_embeddings = self.item_emb(item_ids)
        torch._assert(
            len(item_embeddings.size()) == 3, "len(item_embeddings.size()) must be 3"
        )

        return self._ndp_module(
            query_embeddings=query_embeddings,  # (B, query_embedding_dim)
            item_embeddings=item_embeddings,  # (1/B, X, item_embedding_dim)
        )
    
    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        sampled_ids = self.negatives_sampler(       # bs, seq_len, num_neg
            positive_ids=supervision_ids,
            num_to_sample=self._num_to_sample,
        )
        sampled_negative_embeddings = self._l2_norm(self.item_emb(sampled_ids))   # bs, seq_len, num_neg, dim
        
        positive_embeddings = self._l2_norm(supervision_embeddings)
        positive_logits = self.similarity_fn(
            query_embeddings=output_embeddings,  # [B, D] = [N', D]
            item_ids=supervision_ids.unsqueeze(1),  # [N', 1]
            item_embeddings=positive_embeddings.unsqueeze(1),  # [N', D] -> [N', 1, D]
        )
        positive_logits = positive_logits / self._softmax_temperature  # [0]
        sampled_negatives_logits = self.similarity_fn(
            query_embeddings=output_embeddings,  # [N', D]
            item_ids=sampled_ids,  # [N', R]
            item_embeddings=sampled_negative_embeddings,  # [N', R, D]
        )  # [N', R]  # [0]
        sampled_negatives_logits = torch.where(
            supervision_ids.unsqueeze(1) == sampled_ids,  # [N', R]
            -5e4,
            sampled_negatives_logits / self._softmax_temperature,
        )
        return positive_logits, sampled_negatives_logits, supervision_weights
        
    
    def forward(self, batch_user, batch_pos_seq):
        """
        Parameters
        ----------
        batch_user: 1-D LongTensor (batch_size)
        batch_pos_seq : 2-D LongTensor (batch_size, seq_len)

        Returns
        -------
        output : 
            Model output to calculate its loss function
        """
        pos_seq_embed = self.item_emb(batch_pos_seq)    # bs, seq_len, dim
        lengths = (batch_pos_seq > 0).sum(-1).long()   # bs
        seq_embeddings, _ = self.generate_user_embeddings(
            past_lengths=lengths,
            past_ids=batch_pos_seq,
            past_embeddings=pos_seq_embed,
            past_payloads=None,
        )
        ar_mask = batch_pos_seq[:, 1:] != 0
        supervision_weights = ar_mask.float()
        output_embeddings = seq_embeddings[:, :-1, :]
        supervision_embeddings = pos_seq_embed[:, 1:, :]
        supervision_ids = batch_pos_seq[:, 1:]
        jagged_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        jagged_supervision_ids = (
            torch.ops.fbgemm.dense_to_jagged(
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )
        output = self.jagged_forward(
                output_embeddings=torch.ops.fbgemm.dense_to_jagged(
                    output_embeddings,
                    [jagged_id_offsets],
                )[0],
                supervision_ids=jagged_supervision_ids,
                supervision_embeddings=torch.ops.fbgemm.dense_to_jagged(
                    supervision_embeddings,
                    [jagged_id_offsets],
                )[0],
                supervision_weights=torch.ops.fbgemm.dense_to_jagged(
                    supervision_weights.unsqueeze(-1),
                    [jagged_id_offsets],
                )[0].squeeze(1),
            )
        return output
    
    def get_loss(self, output):
        positive_logits, sampled_negatives_logits, supervision_weights = output
        jagged_loss = -F.log_softmax(
            torch.cat([positive_logits, sampled_negatives_logits], dim=1), dim=1
        )[:, 0]
        loss = (jagged_loss * supervision_weights).sum() / supervision_weights.sum()
        return loss

    def get_current_embeddings(
        self,
        lengths: torch.Tensor,
        encoded_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            lengths: (B,) x int
            seq_embeddings: (B, N, D,) x float

        Returns:
            (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
        """
        B, N, D = encoded_embeddings.size()
        flattened_offsets = (lengths - 1) + torch.arange(
            start=0, end=B, step=1, dtype=lengths.dtype, device=lengths.device
        ) * N
        return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache: Optional[List[HSTUCacheState]],
        return_cache_states: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Args:
            past_lengths: (B,) x int64.
            past_ids: (B, N,) x int64.
            past_embeddings: (B, N, D,) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).
            return_cache_states: bool.

        Returns:
            (B, D) x float, representing embeddings for the current state.
        """
        encoded_seq_embeddings, cache_states = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )  # [B, N, D]
        current_embeddings = self.get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Runs encoder to obtain the current hidden states.

        Args:
            past_lengths: (B,) x int.
            past_ids: (B, N,) x int.
            past_embeddings: (B, N, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            (B, D,) x float, representing encoded states at the most recent time step.
        """
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )

    def get_ratings(self, batch_user):
        past_ids = self.train_seq[batch_user].cuda()
        past_lengths = (past_ids > 0).sum(-1).long()
        shared_input_embeddings = self.encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            past_embeddings=self.item_emb(past_ids),
            past_payloads=None,
        )
        items = self._l2_norm(self.item_emb(self.item_list))
        score_mat = torch.matmul(shared_input_embeddings, items.T)
        return score_mat
    
    def forward_multi_items(self, batch_user, batch_items):
        past_ids = self.train_seq[batch_user].cuda()
        past_lengths = (past_ids > 0).sum(-1).long()
        shared_input_embeddings = self.encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=self.item_emb(past_ids),
            past_payloads=None,
        )
        # change start value of item IDs to 1
        items = self._l2_norm(self.item_emb(batch_items + 1))
        score_mat = torch.bmm(items, shared_input_embeddings.unsqueeze(-1)).squeeze(-1)
        return score_mat

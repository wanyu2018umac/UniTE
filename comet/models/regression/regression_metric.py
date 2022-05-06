# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
RegressionMetric
========================
    Regression Metric that learns to predict a quality assessment by looking
    at source, translation and reference.
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from comet.models.base import CometModel
from comet.modules import FeedForward
from torchmetrics import MetricCollection, PearsonCorrcoef, SpearmanCorrcoef
from transformers import AdamW

from torch.nn.utils.rnn import pad_sequence
from torch.nn import Parameter
from math import ceil
# torch.set_printoptions(edgeitems=100)
# torch.set_printoptions(linewidth=100)


class RegressionMetric(CometModel):
    """RegressionMetric:

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the encoder frozen during training.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg', 'avg_each', 'cls_each'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param hidden_sizes: Hidden sizes for the Feed Forward regression.
    :param activations: Feed Forward activation function.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    :param input_segments: Which segments for input, any combination among 'src', 'hyp' and 'ref'.
    :param pooling_rep: Which representations for regression, any combination among 'hyp', 'ref', 'src_hyp_prod', 'src_hyp_l1', 'ref_hyp_prod' and 'ref_hyp_l1'.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        embedding_learning_rate: Union[None, float] = None,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-base",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[str] = None,
        validation_data: Optional[str] = None,
        hidden_sizes: List[int] = [2304, 768],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        load_weights_from_checkpoint: Optional[str] = None,
        input_segments: List[str] = ['hyp', 'src', 'ref'],
        input_segments_sampling: Dict[str, float] = {},
        pooling_rep: List[str] = ['hyp', 'ref', 'src_hyp_prod', 'src_hyp_l1', 'ref_hyp_prod', 'ref_hyp_l1'],
        combine_inputs: bool = False,
        multiple_segment_embedding: bool = False,
        attention_excluded_regions: List[str] = [],
        attention_excluded_regions_dict: Dict[str, str] = {},
        attention_excluded_regions_sampling: Dict[str, float] = {},
        cls_from_all_to_cls: bool = False,
        cls_from_cls_to_all: bool = False,
        reset_position_for_each_segment: bool = False,
        bos_for_segments: List[str] = ['<s>', '</s>', '</s>'],
        eos_for_segments: List[str] = ['</s>', '</s>', '</s>']
    ) -> None:
        super().__init__(
            nr_frozen_epochs,
            keep_embeddings_frozen,
            optimizer,
            encoder_learning_rate,
            learning_rate,
            layerwise_decay,
            encoder_model,
            pretrained_model,
            pool,
            layer,
            dropout,
            batch_size,
            train_data,
            validation_data,
            load_weights_from_checkpoint,
            "regression_metric",
        )
        self.save_hyperparameters()

        if any(x not in ['hyp', 'src', 'ref'] for x in input_segments):
            raise ValueError('Invalid setting for input segments excluding \'hyp\', \'src\' and \'ref\'.')

        self.input_segments = list()
        for x in ['hyp', 'src', 'ref']:
            if x in input_segments:
                self.input_segments.append(x)
        self.input_segments_dict = dict((v, k) for k, v in enumerate(self.input_segments))
        
        self.input_segments_sampling = list()
        if len(input_segments_sampling) > 0:
            if sum(input_segments_sampling.values()) != 1:
                raise ValueError('Summation of sampling probabilities for input segments must be equal to 1.')
            for k, v in input_segments_sampling.items():
                segments_names = list(x.strip() for x in k.split(','))
                if any(x not in self.input_segments for x in segments_names):
                    raise ValueError('Segment names for sampling must be included in \[%s\]' % ', '.join(self.input_segments))
                self.input_segments_sampling.append((segments_names, v))
            print('Following formatted input segments will be randomly chosen:')
            for k, v in self.input_segments_sampling:
                print('\t%s\t%f' % (', '.join(k), v))

        if 'hyp' in self.input_segments and len(self.input_segments) == 1:
            raise ValueError('Invalid setting for metric because only input is candidate.')
        self.pooling_rep = pooling_rep
        self.combine_inputs = combine_inputs
        self.multiple_segment_embedding = multiple_segment_embedding

        special_tokens = self.encoder.prepare_sample(['<pad>'])['input_ids'].view(-1).cpu().tolist()

        self.pad_idx = special_tokens[1]
        self.bos_idx = special_tokens[0]
        self.eos_idx = special_tokens[2]

        self.special_tokens = {'<s>': self.bos_idx, '<pad>': self.pad_idx, '</s>': self.eos_idx}
        print('Indexes for special tokens:')
        print('PAD (<pad>):', self.pad_idx)
        print('BOS (<s>):', self.bos_idx)
        print('EOS (</s>):', self.eos_idx)

        if self.combine_inputs:
            print('The input will be formatted as the combination of', self.input_segments, '.')
        else:
            print('The input will be seperated as', self.input_segments, '.')
            print('Features for regression: %s' % ' '.join(self.pooling_rep))
        
        if self.multiple_segment_embedding:
            if self.encoder.model.embeddings.token_type_embeddings.weight.size(0) == 1:
                print('Segment embeddings will be repeated for each input among \'[%s]\' for initialization.' % ', '.join(self.input_segments))
                self.encoder.model.embeddings.token_type_embeddings.weight = Parameter(self.encoder.model.embeddings.token_type_embeddings.weight.repeat(len(self.input_segments), 1).contiguous())
            elif self.encoder.model.embeddings.token_type_embeddings.weight.size(0) == len(self.input_segments):
                print('Segment embeddings loaded from checkpoint has the same size as that of training model.')
            else:
                raise ValueError('Segment embeddings loaded from checkpoint can\'t convert to acquired embeddings.')
            print('Segment embeddings size:', self.encoder.model.embeddings.token_type_embeddings.weight.size())

        if len(attention_excluded_regions_sampling) != 0:
            self.attention_excluded_regions_dict = dict()
            # print(attention_excluded_regions_sampling)
            for region_group in attention_excluded_regions_sampling.keys():
                if region_group == "":
                    continue
                if any(x not in ['src-src', 'src-hyp', 'src-ref', 'hyp-src', 'hyp-hyp', 'hyp-ref', 'ref-src', 'ref-hyp', 'ref-ref'] for x in region_group.split(', ')):
                    raise ValueError('Attention excluded regions must be choices from \'src-src, src-hyp, src-ref, hyp-src, hyp-hyp, hyp-ref, ref-src, ref-hyp, ref-ref\'.')
            if sum(attention_excluded_regions_sampling.values()) > 1:
                raise ValueError('Summation of all sampling probabilities shouldn\'t be lower than 1.')
            if any(x <= 0 for x in attention_excluded_regions_sampling.values()):
                raise ValueError('All sampling probabilities should be larger than 0.')

            attention_excluded_regions_sampling_mapping = dict((x, x) for x in attention_excluded_regions_sampling.keys())
            # print(attention_excluded_regions_sampling_mapping)
            segment_input_index = {v: str(k) for k, v in enumerate(self.input_segments)}
            print('Following attention among input segments will be randomly banned:')
            for k, v in attention_excluded_regions_sampling.items():
                print('\t%s\t%f' % (k, v))
                for segment in self.input_segments:
                    attention_excluded_regions_sampling_mapping[k] = attention_excluded_regions_sampling_mapping[k].replace(segment, segment_input_index[segment])
                    # print(attention_excluded_regions_sampling_mapping)
                # input()

            self.attention_excluded_regions_sampling = list()
            for k, v in attention_excluded_regions_sampling_mapping.items():
                if v == "":
                    v = list()
                else:
                    v = list((int(x[0]), int(x[-1])) for x in v.split(', '))
                self.attention_excluded_regions_sampling.append((v, attention_excluded_regions_sampling[k]))
                # print(self.attention_excluded_regions_sampling)
                # input()
        elif len(attention_excluded_regions_dict) != 0:
            self.attention_excluded_regions_sampling = list()
            self.attention_excluded_regions_dict = dict()
            segment_input_index = {v: str(k) for k, v in enumerate(self.input_segments)}
            print('Under the setting of each input formatting, following attention among input segments will be randomly banned:')
            for key, value in attention_excluded_regions_dict.items():
                print('%s: %s' % (key, value))

            for key, value in attention_excluded_regions_dict.items():
                region_group = list(x.strip() for x in value.split(','))
                # print('key:', key)
                # print('value:', value)
                # print('region_group:', region_group)
                if value.strip() == '':
                    self.attention_excluded_regions_dict[key] = list()
                    continue
                if any(x not in ['src-src', 'src-hyp', 'src-ref', 'hyp-src', 'hyp-hyp', 'hyp-ref', 'ref-src', 'ref-hyp', 'ref-ref'] for x in region_group):
                    raise ValueError('Attention excluded regions must be choices from \'src-src, src-hyp, src-ref, hyp-src, hyp-hyp, hyp-ref, ref-src, ref-hyp, ref-ref\'.')

                # print('key:', key)
                # print('value:', value)
                # print('region_group:', region_group)

                for segment in self.input_segments:
                    region_group = list(x.replace(segment, segment_input_index[segment]) for x in region_group)
                    # print('region_group:', region_group)
                    # input()

                self.attention_excluded_regions_dict[key] = list((int(x[0]), int(x[-1])) for x in region_group)
                # print('self.attention_excluded_regions_dict:', self.attention_excluded_regions_dict)
                # input()
  
            print('self.attention_excluded_regions_dict:', self.attention_excluded_regions_dict)
            # input()
            
        else:
            self.attention_excluded_regions_sampling = attention_excluded_regions_sampling
            self.attention_excluded_regions_dict = attention_excluded_regions_dict
            if len(attention_excluded_regions) != 0:
                if any(x not in ['src-src', 'src-hyp', 'src-ref', 'hyp-src', 'hyp-hyp', 'hyp-ref', 'ref-src', 'ref-hyp', 'ref-ref'] for x in attention_excluded_regions):
                    raise ValueError('Attention excluded regions must be choices from \'src-src, src-hyp, src-ref, hyp-src, hyp-hyp, hyp-ref, ref-src, ref-hyp, ref-ref\'.')
                print('Following attention among input segments will be banned: [%s]' % ', '.join(attention_excluded_regions))
                segment_input_index = {v: str(k) for k, v in enumerate(self.input_segments)}
                for segment in self.input_segments:
                    attention_excluded_regions = list(x.replace(segment, segment_input_index[segment]) for x in attention_excluded_regions)
                self.attention_excluded_regions = list((int(x[0]), int(x[-1])) for x in attention_excluded_regions)
                print('Converting attention excluded regions via indexing: [%s]' % ', '.join("%d-%d" % (x[0], x[1]) for x in self.attention_excluded_regions))
            else:
                self.attention_excluded_regions = attention_excluded_regions

        if self.hparams.pool in ['avg_each', 'cls_each'] and not self.combine_inputs:
            raise ValueError('%s pooling only works for setting combine_inputs to True. Please use %s instead.' % (self.hparams.pool, self.hparams.pool[:3]))
            
        self.cls_from_all_to_cls = cls_from_all_to_cls
        self.cls_from_cls_to_all = cls_from_cls_to_all
        self.reset_position_for_each_segment = reset_position_for_each_segment
        if self.reset_position_for_each_segment:
            print('For each segment, position embedding will be restarted from 0.')

        if len(bos_for_segments) < len(self.input_segments):
            raise ValueError('Number of bos for segments doesn\'t match the numebr of input segments.')
        if len(bos_for_segments) > len(self.input_segments):
            print('Number of bos for segments doesn\'t match the number of input segments. Cutting the former to meet the consistency.')
            bos_for_segments = bos_for_segments[:len(self.input_segments)]
        if any(x not in self.special_tokens.keys() for x in bos_for_segments):
            raise ValueError('BOS tokens are advised to be chosen from [%s].' % (', '.join(self.special_tokens.keys())))
        self.bos_for_segments = {k: self.special_tokens[v] for k, v in zip(self.input_segments, bos_for_segments)}
        print('BOS symbols are {%s} for all segments.' % (', '.join(str(x1) + ': ' + str(x2) for (x1, x2) in self.bos_for_segments.items())))

        if len(eos_for_segments) < len(self.input_segments):
            raise ValueError('Number of eos for segments doesn\'t match the numebr of input segments.')
        if len(eos_for_segments) > len(self.input_segments):
            print('Number of bos for segments doesn\'t match the number of input segments. Cutting the former to meet the consistency.')
            eos_for_segments = eos_for_segments[:len(self.input_segments)]
        if any(x not in self.special_tokens.keys() for x in eos_for_segments):
            raise ValueError('EOS tokens are advised to be chosen from [%s].' % (', '.join(self.special_tokens.keys())))
        self.eos_for_segments = {k: self.special_tokens[v] for k, v in zip(self.input_segments, eos_for_segments)}
        print('EOS symbols are {%s} for all segments.' % (', '.join(str(x1) + ': ' + str(x2) for (x1, x2) in self.eos_for_segments.items())))

        in_dim_scale = 1
        if self.combine_inputs and self.hparams.pool in ['avg_each', 'cls_each']:
            in_dim_scale = len(self.input_segments)
        elif not self.combine_inputs:
            in_dim_scale = len(self.pooling_rep)

        self.estimator = FeedForward(
            in_dim=self.encoder.output_units * in_dim_scale,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        return

    def init_metrics(self):
        metrics = MetricCollection(
            {"spearman": SpearmanCorrcoef(), "pearson": PearsonCorrcoef()}
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Sets the optimizers to be used during training."""
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        if self.hparams.embedding_learning_rate is not None:
            layer_parameters[0]['lr'] = self.hparams.embedding_learning_rate

        top_layers_parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]
        if self.layerwise_attention:
            layerwise_attn_params = [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]
            params = layer_parameters + top_layers_parameters + layerwise_attn_params
        else:
            params = layer_parameters + top_layers_parameters

        optimizer = AdamW(
            params,
            lr=self.hparams.learning_rate,
            correct_bias=True,
        )
        # scheduler = self._build_scheduler(optimizer)
        return [optimizer], []

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        
        src_inputs = self.encoder.prepare_sample(sample["src"])
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        ref_inputs = self.encoder.prepare_sample(sample["ref"])
        
        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        inputs = {**src_inputs, **mt_inputs, **ref_inputs}

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def forward(
        self,
        input_segments: str,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt_input_ids: torch.tensor,
        mt_attention_mask: torch.tensor,
        ref_input_ids: torch.tensor,
        ref_attention_mask: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if self.combine_inputs:
            all_input_ids = list()
            # input_segments = input_segments.split('-')
            # print('input_segments:', input_segments)
            inputs_group = input_segments.split('-')
            # input()

            if 'hyp' in inputs_group:
                mt_input_ids[:, 0] = self.bos_for_segments['hyp']
                mt_input_ids_segments = list(x.masked_select(y.ne(0)) for x, y in zip(mt_input_ids.unbind(dim=0), mt_attention_mask.unbind(dim=0)))
                for x in mt_input_ids_segments:
                    x[-1] = self.eos_for_segments['hyp']
                all_input_ids.append(mt_input_ids_segments)

            if 'src' in inputs_group:
                src_input_ids[:, 0] = self.bos_for_segments['src']
                src_input_ids_segments = list(x.masked_select(y.ne(0)) for x, y in zip(src_input_ids.unbind(dim=0), src_attention_mask.unbind(dim=0)))
                for x in src_input_ids_segments:
                    x[-1] = self.eos_for_segments['src']
                all_input_ids.append(src_input_ids_segments)

            if 'ref' in inputs_group:
                ref_input_ids[:, 0] = self.bos_for_segments['ref']
                ref_input_ids_segments = list(x.masked_select(y.ne(0)) for x, y in zip(ref_input_ids.unbind(dim=0), ref_attention_mask.unbind(dim=0)))
                for x in ref_input_ids_segments:
                    x[-1] = self.eos_for_segments['ref']
                all_input_ids.append(ref_input_ids_segments)
            
            # print(all_input_ids)
            # input()

            if len(inputs_group) == 3:
                all_input_concat_padded, all_input_seq_lens = cut_long_sequences3(all_input_ids, 512, self.pad_idx)
            else:
                all_input_concat_padded, all_input_seq_lens = cut_long_sequences2(all_input_ids, 512, self.pad_idx)

            # print('all_input_seq_lens:', all_input_seq_lens, all_input_seq_lens.size())
            # input()
            
            cls_ids = torch.cat((all_input_seq_lens.new_zeros(size=(all_input_seq_lens.size(0), 1)), all_input_seq_lens.cumsum(dim=-1)), dim=-1).contiguous()  # batch_size x (num_of_inputs + 1)
            # print('cls_ids:', cls_ids, cls_ids.size())
            # input()

            token_type_ids, token_type_masks = compute_token_type_ids(
                token_ids=all_input_concat_padded,
                cls_ids_with_sum_lens=cls_ids,
                buffered_position_ids=self.encoder.model.embeddings.position_ids,
                num_of_inputs=len(inputs_group),
                padding_value=len(self.input_segments),
                replaced_seg_ids=list(self.input_segments_dict[x] for x in inputs_group)
            )
            
            all_mask_concat_padded = all_input_concat_padded.ne(self.pad_idx).long()

            if len(self.attention_excluded_regions_sampling) > 0:
                if self.training:
                    sampled_random = torch.rand(size=(1,))
                    regions_group_idx = 0
                    # print('sampled_random, regions_group_idx:', sampled_random, regions_group_idx)
                    # input()
                    while regions_group_idx < len(self.attention_excluded_regions_sampling):
                        regions_group, regions_group_sampling_prob = self.attention_excluded_regions_sampling[regions_group_idx]
                        if sampled_random >= regions_group_sampling_prob:
                            sampled_random -= regions_group_sampling_prob
                            regions_group_idx += 1
                        else:
                            break
                        # print('sampled_random, regions_group_idx:', sampled_random, regions_group_idx)
                        # input()
                    # print('sampled_random, regions_group_idx:', sampled_random, regions_group_idx)
                    # input()

                    excluded_regions_all_mask_concat_padded = compute_attention_masks_for_regions(
                        seq_lens=all_input_seq_lens,
                        excluded_regions=regions_group,
                        buffered_position_ids=self.encoder.model.embeddings.position_ids,
                        num_of_inputs=len(self.input_segments),
                        token_type_ids=token_type_ids,
                        cls_token_ids=cls_ids,
                        cls_from_cls_to_all=self.cls_from_cls_to_all,
                        cls_from_all_to_cls=self.cls_from_all_to_cls
                    )
                    # for x in excluded_regions_all_mask_concat_padded.cpu().tolist():
                    #     print('\n'.join(str(y) for y in x))
                    #     print()
                    # input()
                else:
                    excluded_regions_all_mask_concat_padded = all_mask_concat_padded
            
            elif len(self.attention_excluded_regions_dict) > 0:
                excluded_regions_all_mask_concat_padded = compute_attention_masks_for_regions(
                    seq_lens=all_input_seq_lens,
                    excluded_regions=self.attention_excluded_regions_dict[input_segments],
                    buffered_position_ids=self.encoder.model.embeddings.position_ids,
                    num_of_inputs=len(self.input_segments),
                    token_type_ids=token_type_ids,
                    cls_token_ids=cls_ids,
                    cls_from_cls_to_all=self.cls_from_cls_to_all,
                    cls_from_all_to_cls=self.cls_from_all_to_cls
                )

            elif len(self.attention_excluded_regions) > 0:
                excluded_regions_all_mask_concat_padded = compute_attention_masks_for_regions(
                    seq_lens=all_input_seq_lens,
                    excluded_regions=self.attention_excluded_regions,
                    buffered_position_ids=self.encoder.model.embeddings.position_ids,
                    num_of_inputs=len(self.input_segments),
                    token_type_ids=token_type_ids,
                    cls_token_ids=cls_ids,
                    cls_from_cls_to_all=self.cls_from_cls_to_all,
                    cls_from_all_to_cls=self.cls_from_all_to_cls
                )

            else:
                excluded_regions_all_mask_concat_padded = all_mask_concat_padded
            
            # for x in excluded_regions_all_mask_concat_padded.cpu().tolist():
            #     print('\n'.join(str(y) for y in x))
            #     print()
            #     input()
            
            if self.reset_position_for_each_segment:
                position_ids = compute_position_ids_for_each_segment(all_input_seq_lens, len(self.input_segments), token_type_ids, self.encoder.model.embeddings.position_ids)
            else:
                position_ids = None
            
            # print('inputs_group', inputs_group)
            # print('all_input_concat_padded', all_input_concat_padded)
            # print('excluded_regions_all_mask_concat_padded', excluded_regions_all_mask_concat_padded)
            # print('all_mask_concat_padded', all_mask_concat_padded)
            # print('token_type_ids', token_type_ids)
            # print('token_type_masks', token_type_masks)
            # print('cls_ids', cls_ids)
            # print('position_ids', position_ids)
            # print('len(self.input_segments)', len(self.input_segments))
            # print('input_segments', input_segments)
            # input()

            # print(self.encoder.model.embeddings.word_embeddings.weight, self.encoder.model.embeddings.word_embeddings.weight.size())
            # print(self.encoder.model.embeddings.position_embeddings.weight, self.encoder.model.embeddings.position_embeddings.weight.size())
            # print(self.encoder.model.embeddings.token_type_embeddings.weight, self.encoder.model.embeddings.token_type_embeddings.weight.size())
            # print(self.encoder.model.embeddings.LayerNorm.weight, self.encoder.model.embeddings.LayerNorm.weight.size())
            # print(self.encoder.model.embeddings.LayerNorm.bias, self.encoder.model.embeddings.LayerNorm.bias.size())
            # input()

            embedded_sequences = self.get_sentence_embedding(
                all_input_concat_padded,
                excluded_regions_all_mask_concat_padded,
                all_mask_concat_padded,
                token_type_ids if self.multiple_segment_embedding else None,
                token_type_masks,
                cls_ids[:, :-1],
                position_ids,
                len(input_segments)
            )

        else:
            if 'hyp' in self.input_segments:
                mt_sentemb = self.get_sentence_embedding(mt_input_ids, mt_attention_mask)
            else:
                mt_sentemb = None

            if 'src' in self.input_segments:
                src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
            else:
                src_sentemb = None            
            
            if 'ref' in self.input_segments:
                ref_sentemb = self.get_sentence_embedding(ref_input_ids, ref_attention_mask)
            else:
                ref_sentemb = None

            reps_for_regression = []
            if 'hyp' in self.pooling_rep:
                reps_for_regression.append(mt_sentemb)
            if 'src' in self.pooling_rep:
                reps_for_regression.append(src_sentemb)
            if 'ref' in self.pooling_rep:
                reps_for_regression.append(ref_sentemb)

            if 'ref_hyp_prod' in self.pooling_rep:
                prod_ref = mt_sentemb * ref_sentemb
                reps_for_regression.append(prod_ref)
            
            if 'ref_hyp_l1' in self.pooling_rep:
                diff_ref = torch.abs(mt_sentemb - ref_sentemb)
                reps_for_regression.append(diff_ref)
            
            if 'src_hyp_prod' in self.pooling_rep:
                prod_src = mt_sentemb * src_sentemb
                reps_for_regression.append(prod_src)
            
            if 'src_hyp_l1' in self.pooling_rep:
                diff_src = torch.abs(mt_sentemb - src_sentemb)
                reps_for_regression.append(diff_src)
            
            # diff_ref = torch.abs(mt_sentemb - ref_sentemb)
            # diff_src = torch.abs(mt_sentemb - src_sentemb)

            # prod_ref = mt_sentemb * ref_sentemb
            # prod_src = mt_sentemb * src_sentemb

            embedded_sequences = torch.cat(reps_for_regression, dim=1)

        # print()
        # print(self.encoder.model.embeddings.token_type_embeddings.weight)
        # print(self.encoder.model.embeddings.token_type_embeddings.weight.grad)
        # print()
        return {"score": self.estimator(embedded_sequences)}

    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)

        df = df[["src", "mt", "ref", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["ref"] = df["ref"].astype(str)

        df["score"] = df["score"].astype(float)
        return df.to_dict("records")


# def cut_long_sequences(all_input_concat: List[List[torch.Tensor]], maximum_length: int = 512, eos_idx: int = 2, pad_idx: int = 1):
#     tensor_tuples = list(zip(*all_input_concat))
#     collected_tuples = list()
#     for tensor_tuple in tensor_tuples:
#         if sum(len(x) for x in tensor_tuple) > maximum_length:
#             # print('Lengths:', list(x.size() for x in tensor_tuple))
#             # print(tensor_tuple)
#             offset = ceil((sum(len(x) for x in tensor_tuple) - maximum_length) / len(tensor_tuple))
#             # print('Offset:', offset)
#             new_tensor_tuple = tuple(x[:len(x) - offset] for x in tensor_tuple)
#             for t in new_tensor_tuple:
#                 t[-1].fill_(eos_idx)
#             # print('Modified:', list(x.size() for x in new_tensor_tuple))
#             # print(new_tensor_tuple)
#             collected_tuples.append(new_tensor_tuple)
#             print('Length of combination exceeds %d. Cutting each segment by %d from %d (%s).' % (maximum_length, offset, sum(len(x) for x in tensor_tuple), list(len(x) for x in tensor_tuple)))
#         else:
#             collected_tuples.append(tensor_tuple)
    
#     concat_tensor = list(torch.cat(x, dim=0) for x in collected_tuples)
#     all_input_concat_padded = pad_sequence(concat_tensor, batch_first=True, padding_value=pad_idx)
#     return all_input_concat_padded


def cut_long_sequences2(all_input_concat: List[List[torch.Tensor]], maximum_length: int = 512, pad_idx: int = 1):
    all_input_concat = list(zip(*all_input_concat))
    collected_tuples = list()
    collected_lens = list()
    for tensor_tuple in all_input_concat:
        all_lens = tuple(len(x) for x in tensor_tuple)

        if sum(all_lens) > maximum_length:
            lengths = dict(enumerate(all_lens))
            lengths_sorted_idxes = list(x[0] for x in sorted(lengths.items(), key=lambda d: d[1], reverse=True))

            offset = ceil((sum(lengths.values()) - maximum_length) / 2)

            if min(all_lens) > (maximum_length // 2) and min(all_lens) > offset:
                lengths = dict((k, v - offset) for k, v in lengths.items())
            else:
                lengths[lengths_sorted_idxes[0]] = maximum_length - lengths[lengths_sorted_idxes[1]]

            # new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, list(v for k, v in lengths.items())))
            new_lens = list(lengths[k] for k in range(0, len(tensor_tuple)))
            new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, new_lens))
            for x, y in zip(new_tensor_tuple, tensor_tuple):
                x[-1] = y[-1]
            collected_tuples.append(new_tensor_tuple)
            collected_lens.append(new_lens)
            print('Data length: %d -> %d (%s -> %s).' % (sum(all_lens), sum(lengths.values()), list(all_lens), new_lens))
        else:
            collected_tuples.append(tensor_tuple)
            collected_lens.append(all_lens)

    concat_tensor = list(torch.cat(x, dim=0) for x in collected_tuples)
    all_input_concat_padded = pad_sequence(concat_tensor, batch_first=True, padding_value=pad_idx)
    collected_lens = torch.Tensor(collected_lens).long().to(all_input_concat_padded.device)

    return all_input_concat_padded, collected_lens


def cut_long_sequences3(all_input_concat: List[List[torch.Tensor]], maximum_length: int = 512, pad_idx: int = 1):
    all_input_concat = list(zip(*all_input_concat))
    collected_tuples = list()
    collected_lens = list()
    for tensor_tuple in all_input_concat:
        all_lens = tuple(len(x) for x in tensor_tuple)

        if sum(all_lens) > maximum_length:
            lengths = dict(enumerate(all_lens))
            lengths_sorted_idxes = list(x[0] for x in sorted(lengths.items(), key=lambda d: d[1], reverse=True))

            offset = ceil((sum(lengths.values()) - maximum_length) / 3)

            if min(all_lens) > (maximum_length // 3) and min(all_lens) > offset:
                lengths = dict((k, v - offset) for k, v in lengths.items())
            else:
                while sum(lengths.values()) > maximum_length:
                    if lengths[lengths_sorted_idxes[0]] > lengths[lengths_sorted_idxes[1]]:
                        offset = maximum_length - lengths[lengths_sorted_idxes[1]] - lengths[lengths_sorted_idxes[2]]
                        if offset > lengths[lengths_sorted_idxes[1]]:
                            lengths[lengths_sorted_idxes[0]] = offset
                        else:
                            lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]]
                    elif lengths[lengths_sorted_idxes[0]] == lengths[lengths_sorted_idxes[1]] > lengths[lengths_sorted_idxes[2]]:
                        offset = (maximum_length - lengths[lengths_sorted_idxes[2]]) // 2
                        if offset > lengths[lengths_sorted_idxes[2]]:
                            lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]] = offset
                        else:
                            lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]] = lengths[lengths_sorted_idxes[2]]
                    else:
                        lengths[lengths_sorted_idxes[0]] = lengths[lengths_sorted_idxes[1]] = lengths[lengths_sorted_idxes[2]] = maximum_length // 3

            # new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, list(v for k, v in lengths.items())))
            new_lens = list(lengths[k] for k in range(0, len(lengths)))
            new_tensor_tuple = tuple(x[:y] for x, y in zip(tensor_tuple, new_lens))
            
            for x, y in zip(new_tensor_tuple, tensor_tuple):
                x[-1] = y[-1]
            collected_tuples.append(new_tensor_tuple)
            collected_lens.append(new_lens)
            print('Data length: %d -> %d (%s -> %s).' % (sum(all_lens), sum(lengths.values()), list(all_lens), list(new_lens)))
        else:
            collected_tuples.append(tensor_tuple)
            collected_lens.append(all_lens)

    concat_tensor = list(torch.cat(x, dim=0) for x in collected_tuples)
    all_input_concat_padded = pad_sequence(concat_tensor, batch_first=True, padding_value=pad_idx)
    collected_lens = torch.Tensor(collected_lens).long().to(all_input_concat_padded.device)

    return all_input_concat_padded, collected_lens


def compute_token_type_ids(token_ids: torch.Tensor, cls_ids_with_sum_lens: torch.Tensor, buffered_position_ids: torch.Tensor, num_of_inputs: int, padding_value: int, replaced_seg_ids: List[int]):
    max_seq_lens = token_ids.size(1)
    type_ids_meta = buffered_position_ids[:, :max_seq_lens].expand_as(token_ids)
    type_ids = type_ids_meta.clone().detach().fill_(padding_value)

    # print(seq_lens)
    # print(cumsum_seq_lens)
    # print(type_ids)
    # print('cls_ids_with_sum_lens', cls_ids_with_sum_lens)

    # for i in range(0, num_of_inputs):
    #     sub_seq_lens = cls_ids_with_sum_lens[:, i: i + 1]
    #     # print(sub_seq_lens)
    #     sub_mask = type_ids_meta.lt(sub_seq_lens).long()
    #     # print(sub_mask)
    #     type_ids -= sub_mask
    #     # print(type_ids)

    for i in range(0, num_of_inputs):
        # sub_seq_lens = cls_ids_with_sum_lens[:, i: i + 1]
        # print(sub_seq_lens)
        sub_mask = type_ids_meta.ge(cls_ids_with_sum_lens[:, i: i + 1]) & type_ids_meta.lt(cls_ids_with_sum_lens[:, i + 1: i + 2])
        # print(sub_mask)
        type_ids.masked_fill_(mask=sub_mask, value=replaced_seg_ids[i])
        # print('type_ids', type_ids)
    
    # input()

    mask_for_pad = type_ids.eq(padding_value)
    type_ids_for_out = type_ids.masked_fill(mask=mask_for_pad, value=replaced_seg_ids[-1])

    # print('type_ids_for_out:', type_ids_for_out, type_ids_for_out.size())
    # print('type_ids:', type_ids, type_ids.size())
    # input()
    
    return type_ids_for_out, type_ids


def compute_attention_masks_for_regions(
        seq_lens: torch.Tensor,
        excluded_regions: List[tuple],
        buffered_position_ids: torch.Tensor,
        num_of_inputs: int,
        token_type_ids: torch.Tensor,
        cls_token_ids: torch.Tensor,
        cls_from_cls_to_all: bool,
        cls_from_all_to_cls: bool
        ):
    # print('token_type_ids:', token_type_ids, token_type_ids.size())
    cumsum_seq_lens = seq_lens.cumsum(dim=-1)
    max_seq_lens = int(cumsum_seq_lens[:, -1].max())
    batch_size = seq_lens.size(0)
    type_ids_meta = buffered_position_ids[:,:max_seq_lens].view(1, 1, -1).repeat(batch_size, 1, 1) # batch_size x 1 x max_seq_len
    # sub_mask_meta = type_ids_meta.lt(cumsum_seq_lens[:, -1].view(batch_size, 1, 1)) # batch_size x 1 x 1 -> batch_size x 1 x max_seq_len
    collected_masks = list()
    pivots = cls_token_ids.unsqueeze(dim=-1)  # batch_size x (num_of_inputs + 1) x 1
    # print('Pivots:', pivots, pivots.size())
    for i in range(0, num_of_inputs):
        temp_collected_mask = type_ids_meta.lt(pivots[:, -1:]).long()
        # print('Temp_collected_mask:', temp_collected_mask, temp_collected_mask.size())
        sub_excluded_regions = list(filter(lambda x: x[0] == i, excluded_regions))
        for sub_excluded_region in sub_excluded_regions:
            index = sub_excluded_region[-1]
            temp_mask = (type_ids_meta.ge(pivots[:, index:index + 1]) & type_ids_meta.lt(pivots[:, index + 1:index + 2])).long()
            # print('Temp_mask:', temp_mask, temp_mask.size())
            temp_collected_mask = temp_collected_mask - temp_mask
            # print('Temp_collected_mask:', temp_collected_mask, temp_collected_mask.size())
        collected_masks.append(temp_collected_mask)
        # print()

    collected_masks = torch.cat(collected_masks, dim=1)  # batch_size x num_of_inputs x max_seq_len
    # print('Collected masks:', collected_masks, collected_masks.size())
    # print('Part 1,', token_type_ids, token_type_ids.size())
    # print('Part 2,', buffered_position_ids[:, :batch_size].view(-1, 1) * num_of_inputs, (buffered_position_ids[:, :batch_size].view(-1, 1) * num_of_inputs).size())
    new_idxes = token_type_ids + buffered_position_ids[:, :batch_size].view(-1, 1) * num_of_inputs
    # print('New idxes:', new_idxes, new_idxes.size())
    # token_type_ids: [batch_size, seq_len] -> [batch_size x seq_len]
    # buffered_position_ids: [1, batch_size] -> [batch_size]
    output_masks = collected_masks.view(batch_size * num_of_inputs, -1).index_select(dim=0, index=new_idxes.view(-1)).view(batch_size, max_seq_lens, -1)
    # print('Output masks:', output_masks, output_masks.size())
    # for x in output_masks.cpu().tolist():
    #     print('\n'.join(str(y) for y in x))
    #     print()
    # input()
    if cls_from_cls_to_all:
        output_masks[:, :, 0] = 1
        # print('output_mask after cls_from_cls_to_all:', output_masks)
        # for x in output_masks.cpu().tolist():
        #     print('\n'.join(str(y) for y in x))
        #     print()
        # input()
    if cls_from_all_to_cls:
        temp_mask = type_ids_meta.lt(pivots[:, -1:]).view(batch_size, -1).long()
        output_masks[:, 0, :] = temp_mask
        # print(temp_mask)
        # print('output_mask after cls_from_all_to_cls:')
        # for x in output_masks.cpu().tolist():
        #     print('\n'.join(str(y) for y in x))
        #     print()
        # input()

    return output_masks


def compute_position_ids_for_each_segment(
        seq_lens: torch.Tensor,
        num_of_inputs: int,
        token_type_ids: torch.Tensor,
        buffered_position_ids: torch.Tensor
    ):
    # print('seq_lens', seq_lens)
    cumsum_seq_lens = seq_lens.cumsum(dim=-1) # [batch_size, num_of_inputs]
    # print('cumsum_seq_lens', cumsum_seq_lens)
    max_seq_lens = int(cumsum_seq_lens[:, -1].max())
    batch_size = seq_lens.size(0)
    position_ids = buffered_position_ids[:,:max_seq_lens].repeat(batch_size, 1)  # [batch_size, max_seq_len]
    # print('position_ids', position_ids)
    offset = position_ids.new_zeros(size=position_ids.size(), dtype=position_ids.dtype)
    
    for i in range(1, num_of_inputs):
        temp_mask = position_ids.ge(cumsum_seq_lens[:, i - 1: i]) # [batch_size, num_of_inputs]
        # print('temp_mask', temp_mask)
        temp_offset = temp_mask.long() * seq_lens[:, i - 1: i]
        # print('temp_offset', temp_offset)
        offset = offset + temp_offset
        # print('offset', offset)
        # input()

    position_ids = position_ids - offset
    # print('position_ids', position_ids)
    # input()
    
    return position_ids


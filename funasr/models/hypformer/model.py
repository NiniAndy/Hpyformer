#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import torch
import logging
from torch.cuda.amp import autocast
from typing import Union, Dict, List, Tuple, Optional

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos, rep_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

from funasr.models.hypformer.search import paraformer_greedy_search, paraformer_beam_search
from funasr.models.hypformer.search import  nar2_rescoring, nar_ar_rescoring


@tables.register("model_classes", "Hypformer")
class Hypformer(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """
    
    def __init__(
        self,
        specaug: Optional[str] = None,
        specaug_conf: Optional[Dict] = None,
        normalize: str = None,
        normalize_conf: Optional[Dict] = None,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        decoder: str = None,
        decoder_conf: Optional[Dict] = None,
        hyp_decoder: str = None,
        hyp_decoder_conf: Optional[Dict] = None,
        ctc: str = None,
        ctc_conf: Optional[Dict] = None,
        predictor: str = None,
        predictor_conf: Optional[Dict] = None,
        ctc_weight: float = 0.5,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        # report_cer: bool = True,
        # report_wer: bool = True,
        # sym_space: str = "<space>",
        # sym_blank: str = "<blank>",
        # extract_feats_in_collect_stats: bool = True,
        # predictor=None,
        predictor_weight: float = 0.0,
        predictor_bias: int = 0,
        sampling_ratio: float = 0.2,
        share_embedding: bool = False,
        # preencoder: Optional[AbsPreEncoder] = None,
        # postencoder: Optional[AbsPostEncoder] = None,
        use_1st_decoder_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if decoder is not None:
            decoder_class = tables.decoder_classes.get(decoder)
            decoder = decoder_class(vocab_size=vocab_size, encoder_output_size=encoder_output_size, **decoder_conf,)

        if hyp_decoder is not None:
            hyp_decoder_class = tables.decoder_classes.get(hyp_decoder)
            hyp_decoder = hyp_decoder_class(vocab_size=vocab_size, encoder_output_size=encoder_output_size, **hyp_decoder_conf,)

        if ctc_weight > 0.0:
    
            if ctc_conf is None:
                ctc_conf = {}
            
            ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)
        if predictor is not None:
            predictor_class = tables.predictor_classes.get(predictor)
            predictor = predictor_class(**predictor_conf)
        
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        # self.token_list = token_list.copy()

        # self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.decoder = decoder
        self.hyp_decoder = hyp_decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        #
        # self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
        self.predictor = predictor
        self.predictor_weight = predictor_weight
        self.predictor_bias = predictor_bias
        self.sampling_ratio = sampling_ratio
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
        # self.step_cur = 0

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None
        
        self.use_1st_decoder_loss = use_1st_decoder_loss
        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.error_calculator = None
    
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """

        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        
        batch_size = speech.shape[0]
        stage = kwargs["stage"]

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_ctc, cer_ctc = None, None
        loss_pre = None
        stats = dict()
        
        # decoder: CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(encoder_out, encoder_out_lens, text, text_lengths)
            
            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc
        

        # decoder: Attention decoder branch
        loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att = self._calc_att_loss(encoder_out, encoder_out_lens, text, text_lengths, stage)
        
        loss_att1 = loss_att["loss_att1"]
        loss_att2 = loss_att["loss_att2"]
        loss_att = loss_att1 + loss_att2 

        acc_att1 = acc_att["acc_att1"]
        acc_att2 = acc_att["acc_att2"]

        if cer_att is None:
            cer_att1, cer_att2 = None, None
            wer_att1, wer_att2 = None, None
        else:
            cer_att1 = cer_att["cer_att1"]
            cer_att2 = cer_att["cer_att2"]
            wer_att1 = wer_att["wer_att1"]
            wer_att2 = wer_att["wer_att2"]

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att + loss_pre * self.predictor_weight
        else:
            loss = self.ctc_weight * loss_ctc + (1.2 - self.ctc_weight) * loss_att + loss_pre * self.predictor_weight
        
        # Collect Attn branch stats
        stats["loss_att"] = loss_att2.detach() if loss_att2 is not None else None
        stats["loss_att1"] = loss_att1.detach() if loss_att1 is not None else None
        stats["pre_loss_att"] = pre_loss_att.detach() if pre_loss_att is not None else None
        stats["acc"] = acc_att2
        stats["acc1"] = acc_att1
        stats["cer"] = cer_att2
        stats["cer1"] = cer_att1
        stats["wer"] = wer_att2
        stats["wer1"] = wer_att1
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None
        stats["loss"] = torch.clone(loss.detach())
        
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = (text_lengths + self.predictor_bias).sum()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
    

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):

            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            
            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)
        

        # Forward encoder
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens
    
    def calc_predictor(self, encoder_out, encoder_out_lens):
        
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(
            encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(encoder_out, 
                                                                                       None,
                                                                                       encoder_out_mask,
                                                                                       ignore_id=self.ignore_id)
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index
    
    def cal_decoder_with_predictor(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens):
        
        decoder_outs = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        stage: str,
    ):
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self.predictor(encoder_out, ys_pad, encoder_out_mask,ignore_id=self.ignore_id)
        
        # 0. sampler
        decoder_out_1st = None
        pre_loss_att = None
        if self.sampling_ratio > 0.0:
            sematic_embeds, decoder_out_1st = self.sampler(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds)
        else:
            sematic_embeds = pre_acoustic_embeds
        
        # 1. Forward decoder
        # 1.1 Paraformer decoder
        hpy_outs = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)
        hpy_pro, hpy_mask = hpy_outs[0], hpy_outs[1]
        hpy_mask = hpy_mask.bool()
        hpy_token = hpy_pro.argmax(-1)

        # 1.1.2 error sampler
        if stage == "train":
            hpy_token = self.error_sampler(hpy_token, hpy_pro, hpy_mask, ys_pad)

        hpy_token_pad = hpy_token * hpy_mask + ys_pad * (~hpy_mask)
        hpy_token_pad = rep_eos(hpy_token_pad, self.eos, self.ignore_id)

        # 1.2 Hypformer decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        ## align with the length of hypothesis
        sos_pad = torch.full((hpy_token_pad.size(0), 1), self.sos, dtype=torch.long, device=hpy_token_pad.device)
        hyp_in_pad = torch.cat((sos_pad, hpy_token_pad), dim=-1)
        assert hyp_in_pad.size() == ys_in_pad.size(), "The length of hypothesis is not equal to the length of ys_pad"
        decoder_out, _ = self.hyp_decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, hyp_in_pad)

        if decoder_out_1st is None:
            decoder_out_1st = hpy_pro
        # 2. Compute attention loss
        loss_att1 = self.criterion_att(hpy_pro, ys_pad)
        acc_att1 = th_accuracy(decoder_out_1st.view(-1, self.vocab_size), ys_pad, ignore_label=self.ignore_id,)
        loss_att2 = self.criterion_att(decoder_out, ys_out_pad)
        acc_att2 = th_accuracy(decoder_out.view(-1, self.vocab_size), ys_out_pad, ignore_label=self.ignore_id,)

        loss_att = {"loss_att1": loss_att1, "loss_att2": loss_att2}
        acc_att = {"acc_att1": acc_att1, "acc_att2": acc_att2}

        loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)
        
        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat1 = decoder_out_1st.argmax(dim=-1)
            cer_att1, wer_att1 = self.error_calculator(ys_hat1.cpu(), ys_pad.cpu())
            ys_hat2 = decoder_out.argmax(dim=-1)
            cer_att2, wer_att2 = self.error_calculator(ys_hat2.cpu(), ys_out_pad.cpu())
            cer_att = {"cer_att1": cer_att1, "cer_att2": cer_att2}
            wer_att = {"wer_att1": wer_att1, "wer_att2": wer_att2}
        
        return loss_att, acc_att, cer_att, wer_att, loss_pre, pre_loss_att
    
    def sampler(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds):
        
        tgt_mask = (~make_pad_mask(ys_pad_lens, maxlen=ys_pad_lens.max())[:, :, None]).to(ys_pad.device)
        ys_pad_masked = ys_pad * tgt_mask[:, :, 0]
        if self.share_embedding:
            ys_pad_embed = self.decoder.output_layer.weight[ys_pad_masked]
        else:
            ys_pad_embed = self.decoder.embed(ys_pad_masked)
        with torch.no_grad():
            decoder_outs = self.decoder(encoder_out, encoder_out_lens, pre_acoustic_embeds, ys_pad_lens)
            decoder_out, _ = decoder_outs[0], decoder_outs[1]
            pred_tokens = decoder_out.argmax(-1)
            nonpad_positions = ys_pad.ne(self.ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            same_num = ((pred_tokens == ys_pad) & nonpad_positions).sum(1)
            input_mask = torch.ones_like(nonpad_positions)
            bsz, seq_len = ys_pad.size()
            for li in range(bsz):
                target_num = (((seq_lens[li] - same_num[li].sum()).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    input_mask[li].scatter_(dim=0,
                                            index=torch.randperm(seq_lens[li])[:target_num].to(input_mask.device),
                                            value=0)
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)
        
        sematic_embeds = pre_acoustic_embeds.masked_fill(~input_mask_expand_dim, 0) + ys_pad_embed.masked_fill(
            input_mask_expand_dim, 0)
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    def error_sampler(self, hpy_token, hpy_pro, hpy_mask, ys_pad, max_err_p=0.2):
        B, L = hpy_mask.size()
        device = hpy_mask.device
        # 确定被替换的个数
        hpy_len = hpy_mask.sum(dim=-1)
        num_to_replace = (max_err_p * hpy_len).int()
        # 选择要替换的位置
        replace_mask = torch.zeros_like(hpy_mask, dtype=torch.bool)
        for i in range(B):
            replace_indices = torch.where(hpy_mask[i])[0]
            if len(replace_indices) > 0:
                replace_indices = replace_indices[torch.randperm(len(replace_indices))[:num_to_replace[i]]]
                replace_mask[i, replace_indices] = True

        # 获取 hpy_pro 中每个位置按概率排序的 token 索引
        sorted_indices = hpy_pro.argsort(dim=-1, descending=True)
        # 取第2至第5高的 token 索引
        candidates_indices = sorted_indices[:, :, 1:5]
        # 从第2至第5高的 token 中随机选择一个
        random_choice_indices = torch.randint(0, 4, (B, L), device=device)
        selected_tokens = torch.gather(candidates_indices, 2, random_choice_indices.unsqueeze(-1)).squeeze(-1)
        # 替换被选择的位置的 token
        hpy_token = hpy_token.masked_scatter(replace_mask, selected_tokens[replace_mask])
        return hpy_token


    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        
        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc


    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        # init beamsearch
        meta_data = {}
        if (isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is not None:
                speech_lengths = speech_lengths.squeeze(-1)
            else:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend)
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000)

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        # Encoder
        if kwargs.get("fp16", False):
            speech = speech.half()
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = (predictor_outs[0], predictor_outs[1],  predictor_outs[2], predictor_outs[3])
        peaks = self.forward_cif_peaks(alphas, pre_token_length)
        shot  = peaks > self.predictor.threshold
        shot_index = shot.nonzero()

        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return []

        decoder_outs = self.cal_decoder_with_predictor(encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length)
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]

        results = []
        b, n, d = decoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            nbest_idx = 0
            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"{nbest_idx+1}best_recog"]

            '''parallel decoder greedy search'''
            paraformer_greedy_result = paraformer_greedy_search(decoder_out, ys_pad_lens, peaks)
            paraformer_greedy_result_token_int = paraformer_greedy_result[0].tokens
            output_token_int = list(filter(lambda x: x != self.eos and x != self.sos and x != self.blank_id, paraformer_greedy_result_token_int))

            '''nar2 rescoring'''
            attnre_weight = kwargs.get("decoding_attnre_weight", 0.5)
            beam_size = kwargs.get("beam_size", 10)
            paraformer_beam_result = paraformer_beam_search(decoder_out, ys_pad_lens, beam_size=beam_size, eos=self.eos)
            attn_res_result = nar2_rescoring(self, prefix_results=paraformer_beam_result, encoder_outs=encoder_out,
                                             encoder_lens=torch.tensor([encoder_out.size(1)]), prefix_weight=attnre_weight, reverse_weight=0.0)
            attn_res_token_int = attn_res_result[0].tokens
            output_token_int = list(filter(lambda x: x != self.eos and x != self.sos and x != self.blank_id, attn_res_token_int))

            '''nar_err_ar_decoding'''
            attnre_weight=kwargs.get("decoding_attnre_weight", 0.5)
            beam_size = kwargs.get("beam_size", 10)
            paraformer_beam_result = paraformer_beam_search(decoder_out, ys_pad_lens, beam_size=beam_size, eos=self.eos)
            attn_res_result = nar_ar_rescoring(self, prefix_results=paraformer_beam_result, encoder_outs=encoder_out,
                                               encoder_lens=torch.tensor([encoder_out.size(1)]), prefix_weight=attnre_weight, reverse_weight=0.0)
            attn_res_token_int = attn_res_result[0].tokens
            output_token_int = list(filter(lambda x: x != self.eos and x != self.sos and x != self.blank_id, attn_res_token_int))

            if tokenizer is not None:
                # Change integer-ids to tokens
                token = tokenizer.ids2tokens(output_token_int)
                text_postprocessed = tokenizer.tokens2text(token)
                if not hasattr(tokenizer, "bpemodel"):
                    text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)

                result_i = {"key": key[i], "text": text_postprocessed}

                if ibest_writer is not None:
                    ibest_writer["token"][key[i]] = " ".join(token)
                    # ibest_writer["text"][key[i]] = text
                    ibest_writer["text"][key[i]] = text_postprocessed
            else:
                result_i = {"key": key[i], "token_int": output_token_int}
            results.append(result_i)

        return results, meta_data
    

    

    def forward_cif_peaks(self, alphas: torch.Tensor, token_nums: torch.Tensor) -> torch.Tensor:
        cif2_token_nums = alphas.sum(-1)
        scale_alphas = alphas / (cif2_token_nums / token_nums).unsqueeze(1)
        # peaks = cif_without_hidden(scale_alphas, self.predictor.threshold - 1e-4)
        peaks = cif_without_hidden(scale_alphas, self.predictor.threshold)
        return peaks
    
    def ctc_logprobs(self,
                    encoder_out: torch.Tensor,
                    blank_penalty: float = 0.0,
                    blank_id: int = 0):
        if blank_penalty > 0.0:
            logits = self.ctc.ctc_lo(encoder_out)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:
            ctc_probs = self.ctc.log_softmax(encoder_out)

        return ctc_probs


    
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ar_ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ar_ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ar_ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_out_lens = torch.tensor([encoder_out.size(1)]).repeat(encoder_out.size(0)).to(encoder_out.device)
        decoder_out, _ = self.hyp_decoder.forward_onestep(encoder_out, encoder_out_lens, hyps, hyps_lens)
        r_decoder_out = torch.zeros_like(decoder_out)

        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        return decoder_out, r_decoder_out

    def forward_nar_ar_decoding(
            self,
            hyps: torch.Tensor,
            hyps_lens: torch.Tensor,
            encoder_out: torch.Tensor,
            reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_out_lens = torch.tensor([encoder_out.size(1)]).repeat(encoder_out.size(0)).to(encoder_out.device)
        decoder_out, _ = self.hyp_decoder.forward_nar2(encoder_out, encoder_out_lens, hyps, hyps_lens)
        r_decoder_out = torch.zeros_like(decoder_out)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        return decoder_out, r_decoder_out


def cif_without_hidden(alphas: torch.Tensor, threshold: float):
    # https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/models/predictor/cif.py#L187
    batch_size, len_time = alphas.size()

    # loop varss
    integrate = torch.zeros([batch_size], device=alphas.device)
    # intermediate vars along time
    list_fires = []

    for t in range(len_time):
        alpha = alphas[:, t]

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place, integrate -
            torch.ones([batch_size], device=alphas.device) * threshold,
            integrate)

    fires = torch.stack(list_fires, 1)
    return fires




class DecodeResult:

    def __init__(self,
                 tokens: List[int],
                 score: float = 0.0,
                 confidence: float = 0.0,
                 tokens_confidence: List[float] = None,
                 times: List[int] = None,
                 nbest: List[List[int]] = None,
                 nbest_scores: List[float] = None,
                 nbest_times: List[List[int]] = None):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times

def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def ctc_greedy_search(ctc_probs: torch.Tensor,
                      ctc_lens: torch.Tensor,
                      blank_id: int = 0) -> List[DecodeResult]:
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    
    max_len  = maxlen
    max_len = max_len if max_len > 0 else ctc_lens.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=ctc_lens.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = ctc_lens.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand  # (B, maxlen)
    mask = mask.to(ctc_probs.device)

    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    results = []
    for hyp in hyps:
        r = DecodeResult(remove_duplicates_and_blank(hyp, blank_id))
        results.append(r)
    return results[0].tokens
'''
Search algorithms for ASR refer to WeNet
'''

import math
from typing import List, Optional, Tuple, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

WHISPER_LANGS = None


def remove_duplicates_and_blank(hyp: List[int], blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths)


def subsequent_mask(size: int, device: torch.device = torch.device("cpu"), ) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def mask_finished_preds(pred: torch.Tensor, flag: torch.Tensor, eos: int) -> torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    return pred.masked_fill_(finished, eos)


def mask_finished_scores(score: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])),
                               dim=1)
        finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])),
                             dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def remove_duplicates_and_blank(hyp: List[int], blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int, ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


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



def nar2_rescoring(
        model,
        prefix_results: List[DecodeResult],
        encoder_outs: torch.Tensor,
        encoder_lens: torch.Tensor,
        prefix_weight: float = 0.0,
        reverse_weight: float = 0.0,
        infos: Dict[str, List[str]] = None,
) -> List[DecodeResult]:
    """
        Args:
            prefix_results(List[DecodeResult]): ar_ctc prefix beam search results
    """
    sos, eos = model.sos, model.eos
    device = encoder_outs.device
    assert encoder_outs.shape[0] == len(prefix_results)
    batch_size = encoder_outs.shape[0]
    results = []

    for b in range(batch_size):
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = prefix_results[b].nbest
        prefix_scores = prefix_results[b].nbest_scores
        prefix_tokens_confidence = prefix_results[b].tokens_confidence
        hyps_pad = pad_sequence([torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps], True,
                                model.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps], device=device, dtype=torch.long)  # (beam_size,)

        hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        prefix_len = 1
        decoder_out, r_decoder_out = model.forward_attention_decoder(hyps_pad, hyps_lens, encoder_out, reverse_weight)
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        confidences = []
        tokens_confidences = []
        for i, hyp in enumerate(hyps):
            score = 0.0
            tc = []  # tokens confidences
            for j, w in enumerate(hyp):
                s = decoder_out[i][j + (prefix_len - 1)][w]
                score += s
                tokens_confidences_num = math.exp(s)
                tc.append(tokens_confidences_num)
            score += decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
            # add right to left decoder score
            if reverse_weight > 0 and r_decoder_out.dim() > 0:
                r_score = 0.0
                for j, w in enumerate(hyp):
                    s = r_decoder_out[i][len(hyp) - j - 1 + (prefix_len - 1)][w]
                    r_score += s
                    tc[j] = (tc[j] + math.exp(s)) / 2
                r_score += r_decoder_out[i][len(hyp) + (prefix_len - 1)][eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            confidences.append(math.exp(score / (len(hyp) + 1)))
            # add ar_ctc score
            score += prefix_scores[i] * prefix_weight
            if score > best_score:
                best_score = score
                best_index = i
            tokens_confidences.append(tc)

        best_token_path = hyps[best_index]

        results.append(
            DecodeResult(best_token_path,
                         best_score,
                         confidence=confidences[best_index],
                         times=[],
                         tokens_confidence=tokens_confidences[best_index]))
    return results


def _find_differences_in_tuples(tuples_list):
    differences = {}

    # 获取每个元组的长度，假设所有元组长度相同
    length = len(tuples_list[0])

    # 遍历每个位置
    for i in range(length):
        # 获取该位置所有元组的值
        values_at_position = [tpl[i] for tpl in tuples_list]

        # 如果该位置有不同的值，记录这些差异
        if len(set(values_at_position)) > 1:
            # 用于存储这个位置出现的不同值和对应的元组 ID
            differences[i] = [{val: idx} for idx, val in enumerate(values_at_position)]

    for key, value_list in differences.items():
        seen_keys = set()  # 用于跟踪已经遇到的键
        filtered_list = []  # 用于存储过滤后的列表

        for item in value_list:
            # item 是一个字典，格式 {某个数值: 元组ID}
            for k, v in item.items():
                if k not in seen_keys:
                    seen_keys.add(k)  # 将键添加到已见过的集合中
                    filtered_list.append(item)  # 保留第一次出现的项
                    break  # 跳出当前循环，检查下一个 item

        # 用过滤后的列表替换原来的列表
        differences[key] = filtered_list

    return differences


def nar_ar_rescoring(
        model,
        prefix_results: List[DecodeResult],
        encoder_outs: torch.Tensor,
        encoder_lens: torch.Tensor,
        prefix_weight: float = 0.0,
        reverse_weight: float = 0.0,
        infos: Dict[str, List[str]] = None,
) -> List[DecodeResult]:

    sos, eos = model.sos, model.eos
    device = encoder_outs.device
    assert encoder_outs.shape[0] == len(prefix_results)
    batch_size = encoder_outs.shape[0]
    results = []

    for b in range(batch_size):
        hyps_n_best = prefix_results[b].nbest
        hyps_tokens_confidence = prefix_results[b].tokens_confidence
        err_ids = _find_differences_in_tuples(hyps_n_best)
        encoder_out = encoder_outs[b, :encoder_lens[b], :].unsqueeze(0)
        hyps = [prefix_results[b].tokens]
        hyps_pad = pad_sequence([torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps], True,
                                model.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps], device=device, dtype=torch.long)  # (beam_size,)

        hyps_pad, _ = add_sos_eos(hyps_pad, sos, eos, model.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining

        for n, latent_id_list in err_ids.items():
            latent_prob = []
            latent_id_confidence_list = []
            candidate_tokens = [list(item.keys())[0] for item in latent_id_list]
            for latent_id in latent_id_list:
                each_candidate_confidence = []
                latent_id, m = next(iter(latent_id.items()))
                latent_id_confidence = hyps_tokens_confidence[m][n]
                latent_id_confidence_list.append(latent_id_confidence)
                hyps_pad[0][n + 1] = latent_id
                decoder_out, _ = model.forward_attention_decoder(hyps_pad, hyps_lens, encoder_out, reverse_weight)
                decoder_out = decoder_out[0][n].unsqueeze(0)
                for candidate in candidate_tokens:
                    prob = decoder_out[0][candidate]
                    candidate_confidence = math.exp(prob)
                    each_candidate_confidence.append(candidate_confidence)
                latent_prob.append(each_candidate_confidence)

            latent_prob = torch.tensor(latent_prob)
            latent_prob = latent_prob.sum(dim=0)
            latent_id_confidence = torch.tensor(latent_id_confidence_list)
            latent_id_confidence = latent_id_confidence * latent_prob

            max_index = torch.argmax(latent_id_confidence).item()
            hyp_out = candidate_tokens[max_index]

            if n + 1 < hyps_lens[0]:
                if hyp_out != model.eos:
                    # update hyps_pad
                    hyps_pad[0][n + 1] = hyp_out
                else:
                    break
            else:
                break

        token = hyps_pad[0][1:]
        results.append(
            DecodeResult(tokens=token)
        )
    return results


def paraformer_greedy_search(
        decoder_out: torch.Tensor,
        decoder_out_lens: torch.Tensor,
        cif_peaks: Optional[torch.Tensor] = None) -> List[DecodeResult]:
    batch_size = decoder_out.shape[0]
    maxlen = decoder_out.size(1)
    topk_prob, topk_index = decoder_out.topk(1, dim=2)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    topk_prob = topk_prob.view(batch_size, maxlen)
    results: List[DecodeResult] = []
    topk_index = topk_index.cpu().tolist()
    topk_prob = topk_prob.cpu().tolist()
    decoder_out_lens = decoder_out_lens.cpu().numpy()
    for (i, hyp) in enumerate(topk_index):
        confidence = 0.0
        tokens_confidence = []
        lens = decoder_out_lens[i]
        for logp in topk_prob[i][:lens]:
            tokens_confidence.append(math.exp(logp))
            confidence += logp
        r = DecodeResult(hyp[:lens],
                         tokens_confidence=tokens_confidence,
                         confidence=math.exp(confidence / lens))
        results.append(r)

    if cif_peaks is not None:
        for (b, peaks) in enumerate(cif_peaks):
            result = results[b]
            times = []
            n_token = 0
            for (i, peak) in enumerate(peaks):
                if n_token >= len(result.tokens):
                    break
                if peak > 1 - 1e-4:
                    times.append(i)
                    n_token += 1
            result.times = times
            # assert len(result.times) == len(result.tokens)
    return results


def paraformer_beam_search(
        decoder_out: torch.Tensor,
        decoder_out_lens: torch.Tensor,
        beam_size: int = 10,
        eos: int = -1) -> List[DecodeResult]:
    mask = make_non_pad_mask(decoder_out_lens)
    nbest, nbest_scores, tokens_confidences = _para_beam_search(decoder_out, beam_size=beam_size)
    best = list(nbest[0])
    best_score = nbest_scores[0]
    best_time = []
    nbest_times = []
    indices, _ = _batch_beam_search(decoder_out, mask, beam_size=beam_size, eos=eos)
    best_hyps = indices[:, 0, :].cpu()
    decoder_out_lens = decoder_out_lens.cpu()
    results_original = []
    # TODO(Mddct): scores, times etc
    for (i, hyp) in enumerate(best_hyps.tolist()):
        r = DecodeResult(hyp[:decoder_out_lens.numpy()[i]])
        results_original.append(r)

    results = []
    results.append(
        DecodeResult(
            tokens=best,
            score=best_score,
            times=best_time,
            nbest=nbest,
            nbest_scores=nbest_scores,
            nbest_times=nbest_times,
            tokens_confidence=tokens_confidences)
    )

    return results


def _para_beam_search(decoder_out, beam_size=10):
    confidences = []
    tokens_confidences = []

    seq_len, vocab_size = decoder_out.shape[1], decoder_out.shape[2]
    # 初始步骤
    log_probs, indices = torch.log_softmax(decoder_out[0, 0, :], dim=-1).topk(beam_size)
    # 初始化激活的假设（hypotheses），初始时每个假设只包含一个token
    hypotheses = [(log_prob, [index]) for log_prob, index in zip(log_probs, indices)]

    # 遍历每个时间步
    for t in range(1, seq_len):
        all_candidates = []
        # 扩展每个当前假设
        for log_prob, seq in hypotheses:
            # 计算当前假设下每个可能扩展的概率
            next_log_probs, next_indices = torch.log_softmax(decoder_out[0, t, :], dim=-1).topk(beam_size)
            all_candidates.extend(
                (log_prob + next_log_prob, seq + [next_index])
                for next_log_prob, next_index in zip(next_log_probs, next_indices)
            )

        # 选出新的 beam_size 个最佳假设
        all_candidates.sort(reverse=True, key=lambda x: x[0])
        hypotheses = all_candidates[:beam_size]

    nbest_scores = []
    nbest = []
    for log_prob, seq in hypotheses:
        nbest.append(tuple(item.item() for item in seq))
        nbest_scores.append(log_prob.item())

    for i, hyp in enumerate(nbest):
        score = 0.0
        tc = []  # tokens confidences
        for j, w in enumerate(hyp):
            s = decoder_out[0][j][w]
            tc.append(math.exp(s))
        tokens_confidences.append(tc)

    return nbest, nbest_scores, tokens_confidences


def _batch_beam_search(
        logit: torch.Tensor,
        masks: torch.Tensor,
        beam_size: int = 10,
        eos: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Perform batch beam search

        Args:
            logit: shape (batch_size, seq_length, vocab_size)
            masks: shape (batch_size, seq_length)
            beam_size: beam size

        Returns:
            indices: shape (batch_size, beam_size, seq_length)
            log_prob: shape (batch_size, beam_size)

        """

    batch_size, seq_length, vocab_size = logit.shape
    masks = ~masks
    # beam search
    with torch.no_grad():
        # b,t,v
        log_post = torch.nn.functional.log_softmax(logit, dim=-1)
        # b,k
        log_prob, indices = log_post[:, 0, :].topk(beam_size, sorted=True)
        end_flag = torch.eq(masks[:, 0], 1).view(-1, 1)
        # mask predictor and scores if end
        log_prob = mask_finished_scores(log_prob, end_flag)
        indices = mask_finished_preds(indices, end_flag, eos)
        # b,k,1
        indices = indices.unsqueeze(-1)

        for i in range(1, seq_length):
            # b,v
            scores = mask_finished_scores(log_post[:, i, :], end_flag)
            # b,v -> b,k,v
            topk_scores = scores.unsqueeze(1).repeat(1, beam_size, 1)
            # b,k,1 + b,k,v -> b,k,v
            top_k_logp = log_prob.unsqueeze(-1) + topk_scores

            # b,k,v -> b,k*v -> b,k
            log_prob, top_k_index = top_k_logp.view(batch_size, -1).topk(beam_size, sorted=True)

            index = mask_finished_preds(top_k_index, end_flag, eos)

            indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)

            end_flag = torch.eq(masks[:, i], 1).view(-1, 1)

        indices = torch.fmod(indices, vocab_size)

    return indices, log_prob

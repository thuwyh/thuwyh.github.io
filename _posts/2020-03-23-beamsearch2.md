---
layout: single
title: 十分钟读懂beam search-2
categories:
    - 深度学习
tags:
    - NLP
    - 文本生成
    - "beam search"
comments: true
toc: true
header:
    teaser: /assets/beamsearch/封面.png 
---
在上一篇文章中我们介绍了基础版的beam search，这篇文章是对它的一个扩展，可以在模型不改的情况下获得更好的生成结果。今天的介绍围绕的也是一篇蛮新的论文，[《The Curious Case of Neural Text Degeneration》](https://arxiv.org/abs/1904.09751 "The Curious Case of Neural Text Degeneration")，根据这篇论文的版面内容，它应该已经被ICLR 2020接收了。

![论文截图](paper.png)

## Beam Search的问题

先解释以下什么要对Beam Search进行改进。因为Beam Search虽然比贪心有所改进，但还是会生成出空洞、重复、前后矛盾的文本。如果你有文本生成经验，一定对这些现象并不陌生。在语言模型还不像如今的BERT、GPT这么厉害的时候，这种现象更加明显。

没有经验也没关系，我们来看一个论文里面的例子。输入模型的引文（context)
> "The study, published in the Proceedings of the They were cattle called Bolivian Cavalleros; they live in a National Academy of Sciences of the United States of remote desert uninterrupted by town, and they speak huge, America (PNAS), was conducted by researchers from the
beautiful, paradisiacal Bolivian linguistic thing. They say,
Universidad Nacional Autónoma de México (UNAM) and

GPT-2模型， Beam Search, num_beams=32的生成结果
>'Lunch, marge.' They don't tell what the lunch is," director
the Universidad Nacional Autónoma de México
Professor Chuperas Omwell told Sky News. "They've only
(UNAM/Universidad Nacional Autónoma de
been talking to scientists, like we're being interviewed by TV
México/Universidad Nacional Autónoma de
reporters. We don't even stick around to be interviewed by
México/Universidad Nacional Autónoma de
TV reporters. Maybe that's how they figured out that they're
México/Universidad Nacional Autónoma de ...”

可以发现即使是如今最顶级的语言模型加上足够长的引文输入，还是无法得到高质量的生成结果。

论文认为这种问题是由于这种试图最大化序列条件概率的解码策略从根上就有问题。他对比了给定同样引文的情况下人类续写和机器生成的词用语言模型计算出来的概率。如下图所示，人类选择的词（橙线）的概率并不是像机器选择的（蓝线）那样总是那些条件概率最大的词。从生成的结果也可以看出，机器生成的结果有大量重复。

![概率对比图](probability.png)

## 解决对策
人们其实尝试了各种办法对Beam Search进行改进，其实都很好理解，这篇论文总结的也比较到位。

### 随机采样
第一种方法是用**随机采样(sampling)**代替取概率最大的词。采样的依据就是解码器输出的词典中每个词的概率分布。相比于按概率“掐尖”，这样会增大所选词的范围，引入更多的随机性。这个方法其实正是我们之前解读过的谷歌开放式聊天机器人Meena采用的方式。当时那篇论文的结论就是这种随机采样的方法远好于Beam Search。但这其实也是有条件的，随机采样容易产生前后不一致的问题。而在开放闲聊领域，生成文本的 **长度都比较短** ，这种问题就被自然的淡化了。

采样的时候有一个可以控制的超参数，称为**温度**(temperature, $T$)。解码器的输出层后面通常会跟一个softmax函数来将输出概率归一化，通过改变$T$可以控制概率的形貌。softmax的公式如下，当$T$大的时候，概率分布趋向平均，随机性增大；当$T$小的时候，概率密度趋向于集中，即强者俞强，随机性降低，会更多地采样出“放之四海而皆准”的词汇。

$$p_i=\frac{\exp(y_i/T)}{\sum \exp(y_i/T)}$$

### top-k采样
这个方法就是在采样前将输出的概率分布截断，取出概率最大的k个词构成一个集合，然后将这个子集词的概率再归一化，最后重新的概率分布中采样词汇。这个办法据说可以获得比Beam Search好很多的效果，但也有一个问题，就是这个k不太好选。

>While top-k sampling leads to considerably higher quality text than either beam search or sampling from the full distribution, the use of a constant k is sub-optimal across varying contexts.

为啥呢？因为这个概率分布变化比较大，有时候可能很均匀(flat)，有的时候比较集中(peaked)。对于集中的情况还好说，当分布均匀时，一个较小的k容易丢掉很多优质候选词。但如果k定的太大，这个方法又会退化回普通采样。

![两种分布，左边是均匀的，右边是集中的](distribution.png)

### 核采样（Nucleus sampling)
首先表示我不确定这个翻译是不是对的。

这是这篇论文提出的方式，也是相比前面那些都更好的采样方式，他不再取一个固定的k，而是固定候选集合的概率密度和在整个概率分布中的比例。也就是构造一个**最小**候选集$V$，使得

$$ \sum_{x \in V}P(x)>p$$

选出来这个集合之后也和top-k采样一样，重新归一化集合内词的概率，并把集合外词的概率设为0。这种方式也称为top-p采样。

论文有一个图，对比了这几种采样方式的效果。

![效果对比图，红字是前后不符，蓝字是重复。Nucleus效果拔群。](sample.png)

### 惩罚重复
为了解决重复问题，还有可以通过**惩罚因子**将出现过词的概率变小或者**强制不使用重复词**来解决。惩罚因子来自于同样广为流传的[《CTRL: A Conditional Transformer Language Model for Controllable Generation》](https://arxiv.org/abs/1909.05858 "CTRL: A Conditional Transformer Language Model for Controllable Generation")。如果大家感兴趣的话后面可以专门写一期可控文本生成方向的解读。

## 代码解析
其实上述各种采样方式在HuggingFace的库里都已经实现了（感动！），我们来看一下代码。

先看top-k和top-p采样

```python
# 代码输入的是logits，而且考虑很周全（我感觉漏了考虑k和p都给了的情况，这应该是不合适的）
# 巧妙地使用了torch.cumsum
# 避免了一个词都选不出来的尴尬情况
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
```

再看看重复惩罚
```python
# 输入的同样是logits(lprobs)
# 同时输入了之前出现过的词以及惩罚系数（大于1的）
# 考虑到了logit是正和负时处理方式应该不一样
def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty
```

最后是重复词去除
```python
# 这个函数将会返回一个不可使用的词表
# 生成n-gram的巧妙方式大家可以借鉴一下
# 下面是一个3-gram的例子
# a = [1,2,3,4,5]
# for ngram in zip(*[a[i:] for i in range(3)]):
#    print(ngram)
def calc_banned_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    # Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].numpy().tolist()
        generated_ngram = generated_ngrams[idx]
        # 就是这巧妙的一句
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy().tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens
```

以上这些代码应该在哪里调用相信看过昨天文章的朋友都应该知道了，这里就放出来最核心的差异。

```python
if do_sample:
    # 这是今天的采样方式
    _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
    # Top-p/top-k filtering，这一步重建了候选集
    _scores = top_k_top_p_filtering(
        _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
    )  # (batch_size * num_beams, vocab_size)
    # re-organize to group the beam together to sample from all beam_idxs
    _scores = _scores.contiguous().view(
        batch_size, num_beams * vocab_size
    )  # (batch_size, num_beams * vocab_size)

    # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
    probs = F.softmax(_scores, dim=-1)
    # 采样
    next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
    # Compute next scores
    next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
    # sort the sampled vector to make sure that the first num_beams samples are the best
    next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
    next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)
else:
    # 这是昨天的beam search方式
    # 直接将log概率相加求条件概率
    next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

    # re-organize to group the beam together (we are keeping top hypothesis accross beams)
    next_scores = next_scores.view(
        batch_size, num_beams * vocab_size
    )  # (batch_size, num_beams * vocab_size)

    next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
```

OK，今天的文章就到这咯，祝大家生成出高质量的文本！
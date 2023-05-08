---
layout: single
title: "Vicuna初体验"
categories: 
    - 深度学习
    - Review
tags: 
    - Vicuna
    - LLMs
    - LMSYS
    - FastChat
comments: true
toc: true
header:
    teaser: /assets/vicuna/about.png
---

今天深入体验了下Vicuna,以下是我的takeaways:

- 指令跟随的能力跟ChatGPT有点差距。最典型的就是下面的身份设定任务都经常失败（如下图）。模型会非常倔强地回复你他是Vicuna，是LMSYS训练的模型。

![](/assets/vicuna/identity.png)

- 针对上面的问题我看了下代码，发现他们专门搞了好几个问身份的语料来训练模型图片，真的是把身份感刻在了骨子里。
- fastchat迭代挺快的，今天试了下他们新加的API功能。整个使用体验几乎和openai的client一模一样，学习成本很低。但目前文档没怎么跟上，有时需要看看代码。例如我在异步环境里用chatCompletion.create失败，看代码才知道要用acreate。
- 试了下Vicuna-7b的embedding，能力非常一般，而且维度4096太大了，那算相似度可真费劲，而且在检索任务上被768维的Instructor Embedding秒杀了。
- 看了下lmsys的成员，好家伙，几乎全是中国人，感觉人才这块可能对于中文大模型不会是短板。

![](/assets/vicuna/about.png)


- 使用下来总体还可以，下面这个例子和GPT的能力确实差不多。最后一个图是我提供些knowledge给它后的回答，措辞稍微不达预期。

![](/assets/vicuna/compare1.png)
![](/assets/vicuna/compare2.png)
![](/assets/vicuna/knowledge.jpeg)
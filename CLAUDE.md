# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Hugo static site blog (多头注意力 / Multi-Head Attention) using the PaperMod theme. The site is deployed to GitHub Pages via GitHub Actions.

## Commands

```bash
# Local development server
hugo server -D

# Build for production
hugo --minify

# Create a new post
hugo new post/YYYY-MM-DD-post-name.md
```

## Content Structure

- Posts are in `content/post/` with filename pattern `YYYY-MM-DD-title.md`
- Posts can be organized in subdirectories by category (e.g., `ML101/`, `Review/`, `Thoughts/`, `deeplearning/`, `金融/`, `AI-weekly/`)
- Static assets go in `static/assets/` - reference them as `/assets/...` in posts

## Post Front Matter

```yaml
---
title: 'Post Title'
date: "YYYY-MM-DDTHH:MM:SSZ"
draft: false  # Set to true for drafts
categories:
- 深度学习
tags:
- NLP
- ChatGPT
toc: true  # Enable table of contents
---
```

## Math Support

MathJax is enabled globally. Use:
- Inline math: `$...$` or `\(...\)`
- Block math: `$$...$$` or `\[...\]`

## Deployment

- Push to `hugo_src` branch triggers GitHub Actions deployment
- Uses Hugo v0.120.4 extended
- Deploys to GitHub Pages at https://www.yuanhao.site

## Theme Customization

- Theme: PaperMod (git submodule at `themes/PaperMod`)
- Custom partials in `layouts/partials/` (MathJax configuration)
- Custom shortcodes in `layouts/shortcodes/`

## AI-Weekly 写作流程

`content/post/AI-weekly/` 目录下的博客使用以下三阶段采访式写作流程：

### 阶段一：理解提纲
当用户发来博客提纲时：
- 用一两句话复述核心观点，确认理解正确
- 指出提纲中最有潜力展开的 2-3 个点
- 询问用户是否准备好开始采访

### 阶段二：深度采访
采访目标是挖掘写博客需要的「素材」——故事、案例、数据、思考过程。

**采访原则：**
- 每次只问 1 个问题，等用户回答后再追问
- 像播客主持人一样自然对话，不要像填表
- 多用「为什么」「具体是怎样的」「能举个例子吗」
- 如果用户回答得笼统，温和地追问细节
- 适时总结已经聊到的内容，帮用户理清思路

**追问方向：**
- 具体案例："你提到 XX，能讲一个具体的例子吗？"
- 思考过程："你是怎么得出这个结论的？中间有没有想过其他可能？"
- 反面思考："有人可能会反驳说 XX，你怎么看？"
- 情感共鸣："当时是什么感受？"
- 行动建议："如果读者想尝试，你建议他们从哪里开始？"

**采访节奏：**
- 一般进行 5-8 轮追问
- 当某个点已经挖得足够深时，自然过渡到下一个点
- 采访中随时留意可以作为「金句」或「标题」的表达

**结束采访：**
- 总结已收集到的要点
- 询问："还有什么你觉得很重要但我们没聊到的吗？"
- 确认后进入整理阶段

### 阶段三：整理成稿
根据 `content/post/AI-weekly/` 中的历史博客学习用户的写作风格，将采访内容整理成博客。

**结构要求：**
- 开头要抓人，可以用故事、问题或反常识观点切入
- 中间层层递进，每个段落有明确的推进
- 结尾有力，可以是行动号召、开放性问题或金句收尾

**风格要求：**
- 参考历史博客，模仿用户的句式长短、常用词汇、举例方式、段落切分习惯
- 保持口语化但不啰嗦
- 适当保留采访中的原话作为「真实感」来源

**输出格式：**
- 使用 Markdown
- 建议一个标题 + 2-3 个备选标题
- 如果文章较长，提供小标题
- 文末可附上「采访精华」作为写作素材存档

**注意事项：**
- 作为搭档而非工具，可以有自己的观点和建议，但最终决定权在用户
- 如果用户的观点有明显漏洞或争议，采访时温和指出
- 不要自己编造案例或数据，所有内容必须来自采访
- 如果用户说「跳过采访直接写」，提醒采访能让文章更有深度，但尊重选择

---
title: '管中窥豹：从OpenManus看到底什么是Agent产品'
date: "2025-03-07T09:02:47-08:00"
draft: false
categories:
- Review
tags:
- Agent
- AI
- LLM
---

前几天号称“东方破晓”的[Manus.im](https://Manus.im)着实火了一把。但这几天的演进也充满戏剧性，已经有一大堆开源项目来复刻他们。[OpenManus](https://github.com/mannaandpoem/OpenManus)算是比较早的一个，号称3小时就复刻了，但其实他们背后的MetaGPT团队已经搞Agent很久了。我感觉目前的产品惊艳或者翻车都很正常，整个行业才刚开始。但OpenManus这样的项目是一个很不错的学习Agent开发的切入点，它足够简单，也覆盖了足够多的东西。这篇文章是我的一个学习笔记外加一些思考，跟大家交流。

事先说明，开源代码库变化很快，我是2025年3月7号下载的，所有内容都基于当时的版本。先来看一下整个工程的结构：

```
.
├── LICENSE
├── README.md
├── README_zh.md
├── app
│   ├── __init__.py
│   ├── agent
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── manus.py
│   │   ├── planning.py
│   │   ├── react.py
│   │   ├── swe.py
│   │   └── toolcall.py
│   ├── config.py
│   ├── exceptions.py
│   ├── flow
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── flow_factory.py
│   │   └── planning.py
│   ├── llm.py
│   ├── logger.py
│   ├── prompt
│   │   ├── __init__.py
│   │   ├── manus.py
│   │   ├── planning.py
│   │   ├── swe.py
│   │   └── toolcall.py
│   ├── schema.py
│   └── tool
│       ├── __init__.py
│       ├── base.py
│       ├── bash.py
│       ├── browser_use_tool.py
│       ├── create_chat_completion.py
│       ├── file_saver.py
│       ├── google_search.py
│       ├── planning.py
│       ├── python_execute.py
│       ├── run.py
│       ├── str_replace_editor.py
│       ├── terminate.py
│       └── tool_collection.py
├── assets
│   ├── community_group_10.jpg
│   └── community_group_9.jpg
├── config
│   └── config.example.toml
├── main.py
├── requirements.txt
├── run_flow.py
└── setup.py

8 directories, 45 files
```

可以说是非常简单的，除掉资源、readme、配置，一共只有30来个文件。不过朋友们可能会想也许是requirements里有很多强大的包干了很多基础性工作，所以来看下工程依赖，我把每个包干嘛用的也写出来：

```
pydantic：使用 Python 类型注解进行数据验证和设置管理的库。
openai：OpenAI API 的客户端库，用于访问 GPT-4 等模型。
tenacity：实现重试逻辑和弹性模式的库。
pyyaml：Python 的 YAML 解析器和发射器，用于配置处理。
loguru：简单优雅的日志库，具有更好的格式化功能。
numpy：Python 科学计算的基础包。
datasets：轻松访问和共享机器学习数据集的库。
html2text：将 HTML 转换为 markdown 格式纯文本的工具。
gymnasium：开发和比较强化学习算法的环境工具包。
pillow：Python 图像处理库的分支，提供图像处理能力。
browsergym：训练 AI 代理使用网络浏览器的环境。
uvicorn：Python 网络应用的 ASGI 服务器实现。
unidiff：解析和操作统一差异文件的库。
browser-use：构建能使用网络浏览器的 AI 代理的框架。
googlesearch-python：无需 API 密钥即可搜索 Google 的库。
aiofiles：asyncio 的文件支持，支持异步文件操作。
pydantic_core：为 Pydantic 提供核心功能的包。
colorama：跨平台彩色终端文本库。
playwright：浏览器自动化库，用于控制 Chrome、Firefox 和 Safari。
```

可以看到并没有啥特别的，主要是一些跟agent工具相关的包。

工程结构也很清晰，分为agent、flow、prompt、tool四个目录，app这个主目录还有一些辅助的类定义。

## Tool
先来看tool这个目录，包含了agent可以使用的工具实现。所有的工具命名都很清晰，实现最复杂的是`planning.py`，`browser_use_tool.py`和`str_replace_editor.py`，各有几百行代码，其他都是很简单的小工具。

看过Manus demo的朋友应该对里面的plan过程印象比较深刻，它会建一个文件，用markdown格式写下一步一步要做什么，然后做完了打勾。这里的planning.py干的就是这事儿，它维护了一个计划的状态。

[browser_use](https://github.com/browser-use/browser-use)本身就是一个很成功的agent项目，让AI可以操作浏览器。根据官网介绍，这个项目获得了yc的投资，团队就2个人。
![Browser use](/assets/bu.gif)

> 说到tool，我准备在后面写一篇文章聊一下对最近蛮火的MCP的理解，大家感兴趣的话可以点个关注。当然，这个项目没有用到MCP。

## Prompt
再来看一下openmanus的prompt，就看两个，一个是planning，prompt如下

```
PLANNING_SYSTEM_PROMPT = """
You are an expert Planning Agent tasked with solving complex problems by creating and managing structured plans.
Your job is:
1. Analyze requests to understand the task scope
2. Create clear, actionable plans with the `planning` tool
3. Execute steps using available tools as needed
4. Track progress and adapt plans dynamically
5. Use `finish` to conclude when the task is complete

Available tools will vary by task but may include:
- `planning`: Create, update, and track plans (commands: create, update, mark_step, etc.)
- `finish`: End the task when complete

Break tasks into logical, sequential steps. Think about dependencies and verification methods.
"""

NEXT_STEP_PROMPT = """
Based on the current state, what's your next step?
Consider:
1. Do you need to create or refine a plan?
2. Are you ready to execute a specific step?
3. Have you completed the task?

Provide reasoning, then select the appropriate tool or action.
"""
```

还有Manus的prompt

```
SYSTEM_PROMPT = "You are OpenManus, an all-capable AI assistant, aimed at solving any task presented by the user. 
You have various tools at your disposal that you can call upon to efficiently complete complex requests. 
Whether it's programming, information retrieval, file processing, or web browsing, you can handle it all."

NEXT_STEP_PROMPT = """You can interact with the computer using PythonExecute, save important content and information files through FileSaver, open browsers with BrowserUseTool, and retrieve information using GoogleSearch.

PythonExecute: Execute Python code to interact with the computer system, data processing, automation tasks, etc.

FileSaver: Save files locally, such as txt, py, html, etc.

BrowserUseTool: Open, browse, and use web browsers.If you open a local HTML file, you must provide the absolute path to the file.

GoogleSearch: Perform web information retrieval

Based on user needs, proactively select the most appropriate tool or combination of tools. For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.
"""
```

都是非常朴素的指令。我想说的是，从prompt这么简单基本上就可以推断出实际的效果并不会很好。

## Agent
再来看agent目录，继承关系是这样的:`BaseAgent`-->`ReActAgent`-->`ToolCallAgent`-->`Manus`，那么在每个层级分别干了啥事儿呢：

- `BaseAgent`：定义了一个agent的基础字段比如name、memory、system prompt等和基础行为，比如执行、判断是否卡住等。
- `ReActAgent`：更细致定义了每一步应该怎么做，遵从论文里的定义先`think`再`act`
- `ToolCallAgent`: 更细致地定义了`think`是要选择工具，`act`里要执行think选择的工具
- `Manus`：没有定义新功能，只是覆盖了ToolCallAgent的system prompt和next step prompt

ReAct是Agent领域较早期的一篇著名论文，当时还没有现在的reasoning model，大家都是通过prompt的方法让模型在干活前先想一想，从而表现出推理的行为。非常像老师教学生怎么做题。而现在的reasoning model是把这一行为内化了，模型会自主地产生思考的过程，然后再进行实际的输出。从各种benchmark来看，推理模型已经大大超过了原来靠prompt的方法，成了一个青出于蓝而胜于蓝的学生。


这里帮大家梳理一下manus的运行流程

1. 用户输入一个需求，调用Manus agent的`run`函数来执行这个需求
2. `run`函数是一个循环，在超过步数限制或得到结果之前不停运行agent的step函数，这个step是在`ReActAgent`里定义的，里面又包括两步
    1. `think`: 执行Manus的SYSTEM_PROMPT，主要就是选工具
    2. `act`: 依次执行think里选的工具，把工具执行的结果放在memory里

仔细看`Manus`agent的实现会发现它其实没有用到planning，只有一些基础工具可供使用。所以这也许是好多人反馈效果差的原因。因为你实际上只是使用了一个两年前就有的ReAct模式的Agent。

```python
class Manus(ToolCallAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends PlanningAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), GoogleSearch(), BrowserUseTool(), FileSaver(), Terminate()
        )
    )
```

## Flow
Readme里有写到有一个unstable的flow版本，这个版本确实应该更强大一些，它是在Manus上再套一层planning，有个planning flow来做agent的调度。

`flow` 文件夹包含了管理 AI 代理如何处理和执行任务的编排层。它定义了执行模式并协调代理之间的交互。

### 基础流程（`base.py`）

`BaseFlow` 类是一个抽象基类，其主要功能包括：

- 管理代理集合（支持单个、列表或字典格式）
- 为每个流程确立“主要代理”的概念
- 提供代理管理方法（获取/添加）
- 定义所有流程必须实现的抽象 `execute` 方法
- 使用 Pydantic 进行数据验证和类型安全检查

### 规划流程（`planning.py`）

`PlanningFlow` 继承自 `BaseFlow`，实现了一种基于规划的执行策略：

- 使用 `PlanningTool` 创建和管理执行计划
- 将用户任务拆分为连续的步骤
- 跟踪每个步骤的状态（未开始、进行中、已完成）
- 管理计划执行的整个生命周期
- 处理计划检索和执行过程中的错误恢复

这个流程的执行步骤如下：
1. `await self._create_initial_plan(input_text)`来创建一个计划。对比一下刚才的Manus Agent，那里只是在每一步想想要哪个工具，并没有做更全面的规划。
2. 按照计划，调用一个agent来执行计划里的每一步。这里他们实际上只有一个agent，就是上面的Manus Agent。一个很明显的扩展就是引入更多专业的agent，然后充分发挥不同agent的特长。
3. 检查计划是否被执行完或者出了问题

这里还有一个不同就是会每一步根据整个plan的状态来动态生成一个agent的input。而在Manus Agent里，输入就是用户说的一句话。

```
step_prompt = f"""
CURRENT PLAN STATUS:
{plan_status}

YOUR CURRENT TASK:
You are now working on step {self.current_step_index}: "{step_text}"

Please execute this step using the appropriate tools. When you're done, provide a summary of what you accomplished.
"""
```

总的来说，**planning像是一个自动写prompt的过程**，就是通过它来给agent更多的上下文信息，和更细致的指令。

## 总结

在我看来，从这个项目里基本已经可以看出现在的Agent开发到底是在干什么

1. 升级模型。这个我相信99.99%的开发者没有能力做，基本是在等大厂升级，然后水涨船高。小部分开发者可以针对里面的具体环节做一些训练，比如挑工具、规划等等。
2. 给模型提供更好的工具，比如信息召回、写代码、用浏览器。工具做得越厚，agent这里就可以做得越薄。工具其实也是一个个agent。browser-use就是这个模式，从工具出发，先卖铲子，再看有没有更大的机会。
3. 用prompt来弥补模型的短板。模型懒惰不肯多想，我明确给你分阶段先想后做；模型关注当下没有大局观，我给你整一个planning环节先谋划谋划。绝大部分开发者应该在这里，但很明显1可能会吃掉3，但只要场景够细，我感觉prompt都还有空间。
4. 找好的showcase，提升颜值。Manus这个项目应该给大家上了一课，好看天然就有传播力，更容易火，更容易成。现阶段指望一个执行成功率非常高的通用Agent还挺难的，多少有点抽卡的成分。

跳出来看，Cursor就是一个把上面四个环节全覆盖了的公司，而且能写代码的Agent意味着巨大的想象空间。

以上就是本文全部内容，欢迎大家多交流。下下周有去GTC的朋友也可以约个线下。
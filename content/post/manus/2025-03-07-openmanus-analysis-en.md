---
title: 'Seeing the Big Picture Through a Narrow Lens: Understanding Agent Products Through OpenManus'
date: "2025-03-07T09:02:47-08:00"
draft: false
categories:
- Review
tags:
- Agent
- AI
- LLM
---

The recent "Dawn of the East" hype around [Manus.im](https://Manus.im) sparked significant attention. However, its evolution over the past few days has been dramatic, with numerous open-source projects attempting to replicate it. [OpenManus](https://github.com/mannaandpoem/OpenManus), one of the earliest clones, claims to have replicated it in just 3 hours, though its parent team MetaGPT has been working on agents for much longer. I believe both impressive performances and failures are normal at this early industry stage. Projects like OpenManus serve as excellent entry points for learning agent development - simple yet comprehensive. This article shares my learning notes and reflections.

Disclaimer: The codebase evolves rapidly. All content is based on the version downloaded on March 7, 2025. Let's first examine the project structure:

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

The project is remarkably simple - excluding resources, READMEs, and configs, there are only about 30 core files. Let's examine the dependencies (with usage notes):

```
pydantic: Data validation/settings management via Python type hints
openai: OpenAI API client for GPT-4 access
tenacity: Retry logic implementation
pyyaml: YAML parser for config handling
loguru: Elegant logging solution
numpy: Fundamental scientific computing package
datasets: ML dataset access library
html2text: HTML-to-markdown converter
gymnasium: RL environment toolkit
pillow: Image processing library
browsergym: Browser automation for AI agents
uvicorn: ASGI web server
unidiff: Unified diff parser
browser-use: Browser automation framework
googlesearch-python: Google search without API keys
aiofiles: Async file operations
pydantic_core: Pydantic core utilities
colorama: Cross-platform colored terminal text
playwright: Browser automation library
```

Nothing particularly unique - mainly agent-related utilities.

The architecture is clearly organized into agent, flow, prompt, and tool directories, with auxiliary classes in app.

## Tool
The tool directory contains agent utilities. The most complex implementations are `planning.py`, `browser_use_tool.py`, and `str_replace_editor.py` (each hundreds of lines), while others are simple utilities.

The planning process from Manus demo (markdown task lists with checkboxes) is handled by `planning.py`, which manages plan states.

[browser-use](https://github.com/browser-use/browser-use) itself is a successful agent project enabling browser automation. According to its official site, this YC-funded project is maintained by just two people.
![Browser use](/assets/bu.gif)

> Regarding tools, I'll later write about the trending MCP concept. Note this project doesn't use MCP.

## Prompt
Let's examine two key prompts. Planning prompt:

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

Manus Agent prompt

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

These simplistic prompts suggest limited practical effectiveness.

## Agent
The agent inheritance hierarchy: `BaseAgent`-->`ReActAgent`-->`ToolCallAgent`-->`Manus`. Key implementations:

- `BaseAgent`: Defines basic fields (name, memory, system prompt) and behaviors (execute, detect stalls)
- `ReActAgent`: Implements step-wise execution following the ReAct paper's think-act pattern
- `ToolCallAgent`: Specializes thinking as tool selection and acting as tool execution
- `Manus`: Simply overrides prompts without new functionality

ReAct (Reasoning and Acting) was an early influential paper in agent research, using prompts to simulate reasoning before action. Modern reasoning models internalize this process, outperforming prompt-based approaches in benchmarks.

Manus execution flow:
1. User input triggers `Manus.run()`
2. The loop runs `step()` until completion:
    - `think()`: Select tools using SYSTEM_PROMPT
    - `act()`: Execute selected tools and store results in memory

Notably, `Manus` doesn't utilize planning, only basic tools. This likely explains many user complaints about effectiveness - it essentially uses a two-year-old ReAct pattern agent.


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
The readme mentions an unstable flow version that adds planning layer:

flow directory contains orchestration layers for task handling:

### `base.py`

BaseFlow abstract class features:

- Agent management (single/list/dict)
- "Main agent" concept
- Agent management methods
- Abstract execute method
- Pydantic validation

### `planning.py`
PlanningFlow implements planning-based execution:

- Uses PlanningTool for plan management
- Breaks tasks into sequential steps
- Tracks step states (pending/in-progress/done)
- Manages plan lifecycle
- Handles error recovery

Execution steps:

1. `await self._create_initial_plan(input_text)` creates initial plan
2. Executes plan steps using Manus agent
3. Monitors plan completion/errors

Key difference: Generates dynamic prompts based on plan state, unlike Manus' static input.
```
step_prompt = f"""
CURRENT PLAN STATUS:
{plan_status}

YOUR CURRENT TASK:
You are now working on step {self.current_step_index}: "{step_text}"

Please execute this step using the appropriate tools. When you're done, provide a summary of what you accomplished.
"""
```

Essentially, planning acts as automated prompt engineering, providing richer context and detailed instructions.

## Conclusion
From this project, we observe current agent development focuses on:

- Model Advancement: Mostly driven by tech giants (99.99% developers can't compete)
- Better Tools: Thicker tools enable thinner agents (Browser-use exemplifies this "selling shovels" approach)
- Prompt Engineering: Compensating for model limitations through:
    - Phased thinking/acting prompts
    - Planning phases for big-picture awareness
    - Domain-specific prompt optimizations
- Showcase Crafting: Aesthetic presentation drives virality (Manus demonstrated this well). Reliable general agents remain challenging.

Notably, Cursor exemplifies comprehensive integration of all these aspects, with code-writing agents offering immense potential.

This concludes my analysis. Feel free to discuss further. For those attending GTC next week, let's connect offline!
---
title: 'Reinforcement Learning: From Zero to GRPO'
date: "2025-02-23T21:26:40-08:00"
draft: true
categories:
- RL
tags:
- Reinforcement Learning
- RL
- Machine Learning
---

之前对LLM里的强化学习一直没学透，最近准备把它彻底搞懂，并把我的学习笔记分享出来。这是本系列的第一篇（一共应该也就两三篇），主要涉及一些基本的概念，会以问答对的形式来呈现。

Below is a detailed explanation addressing each of your questions:

---

### 1. Basic Setup of a Reinforcement Learning System

A reinforcement learning (RL) system is composed of several key components:

- **Agent:** The learner or decision-maker that interacts with the environment.
- **Environment:** The external system with which the agent interacts. It provides feedback in the form of states and rewards.
- **State:** A representation of the current situation of the environment.
- **Action:** The set of all possible moves or decisions the agent can take.
- **Reward:** A scalar feedback signal received after taking an action; it tells the agent how good or bad that action was.
- **Policy:** A strategy or mapping from states to actions that the agent follows.
- **Value Function:** An estimate of how good it is to be in a given state (or how good a specific action is in that state), in terms of future cumulative reward.
- **Transition Dynamics:** The rules that determine how the state changes in response to the agent's actions.

In operation, the agent observes the state, takes an action based on its policy, receives a reward, and the environment transitions to a new state. This loop continues, and over time the agent learns to improve its policy to maximize cumulative rewards.

---

### 2. The Goal of Reinforcement Learning

The primary goal in reinforcement learning is to develop a policy that **maximizes the expected cumulative reward** (often referred to as return) over time. Instead of learning from labeled examples as in supervised learning, the agent learns from the consequences of its actions, gradually improving its decision-making strategy based on feedback received from the environment.

---

### 3. Policy-Based Methods

**Policy-based methods** focus on directly optimizing the policy—the mapping from states to actions. Instead of first estimating value functions and then deriving a policy from them, these methods:

- Parameterize the policy directly (e.g., using a neural network that outputs a probability distribution over actions).
- Use gradient ascent techniques to maximize the expected reward by adjusting the policy parameters.
- Examples include algorithms like REINFORCE and Actor-Critic methods, where the “actor” updates the policy and the “critic” estimates the value function to reduce variance in the policy gradient.

---

### 4. Major Drawbacks of Policy-Based Methods

While policy-based approaches have benefits such as naturally handling continuous action spaces and directly optimizing performance, they also come with challenges:

- **High Variance in Gradient Estimates:** The stochastic nature of policy gradients can lead to noisy updates, making the learning process less stable.
- **Sample Inefficiency:** These methods often require a large number of interactions with the environment to obtain accurate gradient estimates.
- **Local Optima:** Because they perform gradient ascent on a non-convex objective, they can get stuck in suboptimal policies.
- **Sensitivity to Hyperparameters:** Learning rates, baseline choices, and other hyperparameters significantly affect performance, necessitating careful tuning.

---

### 5. Value-Based Methods

**Value-based methods** focus on estimating the value of being in a given state or taking a certain action:

- **Core Idea:** They aim to learn a value function (such as a state-value function or an action-value function) that predicts the expected cumulative reward.
- **Deriving Policy:** Once the value function is estimated, a policy can be derived by selecting actions that maximize the expected value. A common approach is to use Q-learning or Deep Q-Networks (DQN), where the agent selects actions with the highest estimated Q-value.

---

### 6. Major Drawbacks of Value-Based Methods

Value-based approaches also face significant challenges:

- **Instability with Function Approximation:** When using approximators like neural networks, value-based methods can suffer from divergence and instability. Techniques like experience replay and target networks are often necessary to stabilize learning.
- **Limited to Discrete Actions:** Many value-based methods are naturally suited for discrete action spaces; adapting them to continuous spaces can be nontrivial.
- **Overestimation Bias:** There is a tendency to overestimate action values, which can lead to suboptimal policies if not corrected.
- **Sensitivity to Hyperparameters and Environment Dynamics:** The methods can be quite sensitive to how the value function is updated and the underlying dynamics of the environment, sometimes requiring extensive tuning and regularization.

---

This overview provides a comprehensive breakdown of the fundamental concepts and trade-offs involved in reinforcement learning systems, especially when comparing policy-based and value-based approaches.
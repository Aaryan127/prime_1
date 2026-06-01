# Week 1: Foundations

# Task 1: Convolutional Neural Networks

## Reading

### Primary Resource

* Deep Residual Learning for Image Recognition (ResNet)

  * https://arxiv.org/abs/1512.03385

### Supporting Resources
* Dive Into Deep Learning (CNN Chapter)

  * https://d2l.ai/

---

## Implementation

Implement ResNet-18 and train it on CIFAR-10.

You may use:

* PyTorch
* torchvision datasets

The network architecture should be implemented by you rather than imported directly from torchvision.models.

---

## Expected Deliverables

1. Training code
2. Training and validation curves
3. Final test accuracy
4. One-page summary explaining:

   * Residual connections
   * Vanishing gradients
   * Why ResNets work

---

# Task 2: Transformers and Attention

## Reading

### Primary Paper

* Attention Is All You Need

  * https://arxiv.org/abs/1706.03762

Read Sections:

* Introduction
* Scaled Dot-Product Attention
* Multi-Head Attention
* Transformer Architecture

Focus on understanding the attention mechanism rather than every implementation detail.

---

## Supporting Resources

* Illustrated Transformer

  * https://jalammar.github.io/illustrated-transformer/

* Harvard Annotated Transformer

  * https://nlp.seas.harvard.edu/annotated-transformer/

* 3blue1Brown Youtube Tutorials on Attention and the Transformer

---

## Expected Deliverable

Write a technical blog post (2–4 pages) explaining:

* Why attention was introduced
* Limitations of RNNs
* Query, Key, and Value vectors
* Scaled Dot-Product Attention
* Multi-Head Attention
* Why transformers became dominant

The blog should target an audience familiar with basic deep learning but new to transformers.

---

# Task 3: Reinforcement Learning Foundations

## Reading

### PPO Paper

* Proximal Policy Optimization Algorithms

  * https://arxiv.org/abs/1707.06347

Read carefully.

### Background Reading

* OpenAI Spinning Up

  * PPO:
    https://spinningup.openai.com/en/latest/algorithms/ppo.html

  * Actor-Critic:
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

---

## Concepts to Understand

* Policy Gradient methods
* Actor-Critic architecture
* Advantage estimation
* PPO clipping objective
* On-policy learning

---

## Implementation

Train an agent on CartPole using PPO.

You may use:

* Stable-Baselines3
* CleanRL
* Gym
The goal is understanding the training pipeline rather than implementing PPO from scratch.

---

## Expected Deliverables

1. Training script
2. Reward curves
3. Trained policy
4. One-page summary explaining:

   * Actor vs Critic
   * Why PPO is stable
   * Why PPO is widely used in robotics

---

# Task 4: Transformer-Based Locomotion for Quadrupeds

This task is intended as preparation for Week 2.

No implementation is required.

---

## Reading

Read and summarize the following papers:

### 1. Decision Transformer

* https://arxiv.org/abs/2106.01345

### 2. Sequence Modeling Solutions for Reinforcement Learning

* https://arxiv.org/abs/2205.14953

### 3. If time permits, read up any recent transformer-based quadruped locomotion paper (2023 onwards)

Suggested search terms:

* Transformer Quadruped Locomotion
* Vision-Language-Action Robotics
* Transformer Reinforcement Learning for Legged Robots
* Foundation Models for Robot Control

---

## Expected Deliverable

Create a 3–5 page literature review covering:

For each paper:

* Problem addressed
* Inputs
* Outputs
* Model architecture
* Advantages
* Limitations

Conclude with:

* How transformers differ from PPO-based policies
* Potential benefits of sequence modeling for quadruped control
* Questions worth exploring in Week 2

---

# Submission Checklist

By the end of Week 1, submit:

* ResNet-18 CIFAR-10 implementation and results
* Transformer attention blog post
* PPO CartPole training results
* Transformer-based locomotion literature review

The emphasis throughout the week should be on understanding concepts clearly for week2.

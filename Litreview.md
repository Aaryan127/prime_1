Literature Review: Decision Transformer and Multi-Agent Transformer
Introduction

Over the past week, I explored two papers that apply transformer architectures to reinforcement learning: Decision Transformer (2021) and Multi-Agent Transformer (2022). Both papers are built on the observation that transformers are extremely effective at modeling long sequences and long-range dependencies. However, they use this idea in different ways.

Decision Transformer treats reinforcement learning as a sequence modeling problem and attempts to predict actions from past trajectories. Multi-Agent Transformer extends transformer-based ideas to multi-agent reinforcement learning while still using traditional reinforcement learning concepts such as policy gradients, critics, and advantages.

The purpose of this review is to summarize my understanding of these papers, discuss their strengths and weaknesses, and reflect on how these ideas could potentially be applied to quadruped robotics.


1. Decision Transformer (2021)

Problem Addressed

Most reinforcement learning algorithms learn through interaction with an environment. They estimate value functions, compute advantages, or optimize policies using rewards. This often leads to complex training procedures involving critics, Bellman equations, and policy updates.

The Decision Transformer paper asks a very different question:

What if reinforcement learning could simply be treated as a sequence prediction problem?

Instead of learning a value function or explicitly maximizing reward, the model learns from previously collected trajectories and predicts actions that successful trajectories would have taken.

This idea is interesting because it completely changes the way reinforcement learning is viewed. Rather than solving an optimization problem, the model learns to imitate successful behavior conditioned on a desired return.

 Inputs

The model is trained on trajectories collected from interactions with an environment.

For every timestep, the sequence contains:

* Return-to-Go (RTG)
* State
* Action

The trajectory is represented as:

(R₁,s₁,a₁,R₂,s₂,a₂,R₃,s₃,a₃,...)

The Return-to-Go represents the amount of reward remaining in the trajectory. One important observation is that RTG is not used as a target. Instead, it is provided as part of the input and acts almost like a goal that the model conditions on.

 Outputs

The output of the model is the next action.

Conceptually, the transformer receives the history of the trajectory and predicts the action that should be taken in the current state while attempting to achieve the specified return.

 Model Architecture

Decision Transformer uses a decoder-only transformer architecture with masked self-attention.

The architecture contains:

* RTG embeddings
* State embeddings
* Action embeddings
* Positional embeddings
* Self-attention layers
* Feed-forward layers
* Action prediction head


When transformers were originally introduced for language modeling, attention helped capture long-range relationships between words in a sentence. For example, a word appearing near the end of a sentence may depend on information that appeared many words earlier.

Decision Transformer applies exactly the same idea to reinforcement learning trajectories.

Imagine an agent picks up a key at timestep 1 and reaches a locked door at timestep 50. The correct decision at timestep 50 depends heavily on something that happened 49 timesteps earlier. Instead of passing information through many recurrent steps, the attention mechanism can directly connect these two events.

This made the transformer feel like a natural choice for reinforcement learning. Rather than learning relationships between words, it learns relationships between states, actions, and rewards across time.


 Advantages

The main advantages observed are:

1. The training process is surprisingly simple because it uses supervised learning rather than traditional reinforcement learning .

2. The model can capture long-range dependencies through attention.

3. No critic, value function, Bellman backup, or advantage estimation is required.

4. The same architecture used in language modeling can be applied directly to decision making.


Limitations

Despite being elegant, the paper has several limitations.

1. The model depends heavily on the quality of the dataset.

2. It cannot easily discover behaviors that do not already exist in the collected trajectories.

3. If the agent enters states that are very different from those seen during training, performance may deteriorate.

4. The model imitates successful behavior but does not explicitly optimize reward.


2. Multi-Agent Transformer (2022)

 Problem Addressed

The Multi-Agent Transformer paper addresses a different challenge: coordination between multiple agents.

In a multi-agent environment, several agents must choose actions simultaneously. The difficulty comes from the fact that the environment receives a joint action containing actions from all agents.

For example:

(a₁,a₂,a₃,...,aₙ)

As the number of agents grows, the size of the joint action space increases dramatically, making coordination difficult.

The key question of the paper is:

Can the coordination problem be reformulated as a sequence modeling problem?

Inputs

The model receives observations from all agents.

During training, the encoder has access to information from the entire system, allowing it to build a global representation of the environment.

I found it useful to think about this using a drone swarm example. Each drone observes a different part of the environment, but the encoder attempts to build a shared understanding of the overall situation.


Outputs

Instead of predicting all actions simultaneously, MAT predicts them sequentially.

Rather than generating:

(a₁,a₂,a₃)

directly,

the model generates:

a₁

then

a₂ conditioned on a₁

then

a₃ conditioned on a₁ and a₂

and so on.

This converts a difficult joint-action problem into a sequence generation problem.

 Model Architecture

Unlike Decision Transformer, MAT uses an encoder-decoder transformer architecture.

The encoder processes observations from all agents and creates a global representation.

The decoder then generates actions one agent at a time.

An important detail is that MAT still contains a critic.

This was one of the biggest differences between the two papers.

Decision Transformer removes most traditional reinforcement learning machinery.

MAT does not.

Instead, it keeps PPO-style training and simply replaces the actor network with a transformer.

The architecture can therefore be viewed as:

Transformer Actor + Critic + PPO-style optimization.

The paper also introduces an advantage decomposition theorem which provides theoretical justification for generating joint actions autoregressively.

My interpretation of this theorem is that it allows credit to be assigned incrementally as each agent's action is generated rather than treating the entire joint action as one indivisible decision.


Advantages

The main advantages are:

1. Better coordination between agents.

2. Ability to model relationships between agents using attention.

3. Scalability through parameter sharing.

4. Compatibility with established reinforcement learning techniques such as PPO.


Limitations

The limitations include:

1. High computational cost due to attention mechanisms.

2. Dependence on an ordering of agents during action generation.

3. Increased architectural complexity compared to standard PPO.

4. Challenges associated with scaling to very large numbers of agents.


How Transformers Differ from PPO-Based Policies

One important conclusion I reached while reading these papers is that transformers and reinforcement learning solve different problems.

Attention determines how information should be processed and which pieces of information are relevant.

Reinforcement learning determines which actions should be encouraged or discouraged.

Decision Transformer attempts to replace reinforcement learning with sequence modeling.

MAT takes a different approach. Instead of replacing reinforcement learning, it uses transformers as a more powerful policy architecture while retaining critics, advantages, and policy-gradient optimization.

This distinction helped me understand why transformer-based methods do not necessarily eliminate the need for reinforcement learning.


Potential Benefits for Quadruped Control

I think transformer-based policies could be particularly useful for quadruped locomotion.

Walking requires coordination between multiple legs over extended periods of time. A decision made several timesteps earlier can affect stability much later in the trajectory.

This resembles the long-range dependency problem that transformers were originally designed to solve.

Additionally, each leg can be viewed as part of a coordinated system. Similar to the way MAT models interactions between multiple agents, transformers may help model interactions between different limbs and joints.

Potential advantages include:

* Better temporal memory
* Improved coordination between limbs
* More robust adaptation to difficult terrain
* Richer representations of locomotion patterns

Questions Worth Exploring Next Week

Based on these papers, I would like to investigate the following questions:

1. Can Decision Transformer be applied directly to quadruped locomotion?

2. Can quadruped legs be treated similarly to agents in MAT?

3. What effect does transformer context length have on locomotion performance?

4. How do transformer-based policies compare with PPO in terms of training efficiency and robustness?


Conclusion

Decision Transformer and Multi-Agent Transformer demonstrate two different ways of combining transformers with reinforcement learning.

Decision Transformer reformulates reinforcement learning as a sequence modeling problem and relies entirely on trajectory prediction. Multi-Agent Transformer retains reinforcement learning while using transformers to improve coordination among multiple agents.

The strongest insight I gained from these papers is that trajectories can be viewed similarly to sentences. In language modeling, attention identifies which previous words are important for predicting the next word. In reinforcement learning, attention identifies which previous states, actions, and rewards are important for making the next decision.

This perspective makes transformers particularly appealing for problems involving long-term dependencies and coordination, which are both central challenges in robotics and locomotion.

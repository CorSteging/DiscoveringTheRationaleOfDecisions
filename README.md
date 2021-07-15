# Discovering the Rationale of Decisions #

This repository contains a set of experiments aimed at evaluating the rationales of machine learning systems in different legal settings.

### Abstract ###
In AI and law, systems that are designed for decision support should be explainable when pursuing justice. In order for these systems to be fair and responsible, they should make correct decisions and make them using a sound and transparent rationale. In this paper, we introduce a knowledge-driven method for model-agnostic rationale evaluation using dedicated test cases, similar to unit-testing in professional software development. We apply this new quantitative human-in-the-loop method in a set of machine learning experiments aimed at extracting known knowledge structures from artificial datasets from a real-life legal setting. We show that our method allows us to analyze the rationale of black box machine learning systems by assessing which rationale elements are learned or not. Furthermore, we show that the rationale can be adjusted using tailor-made training data based on the results of the rationale evaluation. 

### Experiments ###

All three experiments can be found in the following Jupyter notebooks:
* wb_replication.ipynb:  Welfare benefit experiment (Bench-Capon replication)
* wb_simplified.ipynb:   Simplified welfare benefit experiment
* tort_law.ipynb:        Experiment on tort law

### Papers ###

* [Discovering the Rationale of Decisions: Towards a Method for Aligning Learning and Reasoning (ICAIL 2021)](https://www.semanticscholar.org/paper/Discovering-the-Rationale-of-Decisions%3A-Towards-a-Steging-Renooij/73406bf87b403f36935bf375486e3bbbdc69aff2)
* [Discovering the Rationale of Decisions: Experiments on Aligning Learning and Reasoning (XAILA 2021)](https://arxiv.org/abs/2105.06758) 

# AgentOCR_Assignment

This repository is our  course project based on `AgentOCR`, a framework for long-horizon LLM/VLM agents.

## What is AgentOCR?

AgentOCR studies how to make multi-step agents more efficient when they accumulate long interaction histories. Instead of always feeding the full text history back into the model, it converts past observations and actions into compact OCR-readable images. The core idea is that a visual history can preserve useful task context while using fewer context tokens than raw text.

In this project setting, AgentOCR is relevant because it sits at the intersection of sequence modeling, memory compression, and decision making for interactive AI systems. It is also a concrete systems-oriented example of how representation design affects downstream agent performance and efficiency.

## Why We Chose It for course

We chose AgentOCR as our  course project for three main reasons.

1. It addresses a clear data and efficiency problem: long-horizon agents suffer from rapidly growing history length, which makes context management a practical bottleneck.
2. It combines algorithmic ideas with empirical evaluation: the project is not only about model training, but also about how to structure, compress, and reuse sequential information.
3. It is a good fit for course study because it allows us to analyze both effectiveness and efficiency, including task success rate, memory usage, and token cost.

More broadly, this project lets us study how different history representations influence agent behavior on sequential decision-making tasks such as embodied interaction and search-based question answering.

# Temporal Biases in Language Models When Performing QA Tasks

This academic project investigates the performance of the ELECTRA-small model on Question Answering tasks involving temporal reasoning. 

Temporal reasoning refers to the ability to understand and reason about the relationships between events, people, and things over time, oftenexpressed through temporal vocabulary such as "before," "after," and "during."

This repository includes the full project paper as well as some code that visualizes performance and error analysis.

# Summary

## Datasets

**SQuAD Dataset**: Used as the baseline dataset for training and evaluation. It contains over 100,000 question-answer pairs sourced from Wikipedia articles.

**Adversarial Dataset**: Manually created using excerpts from Wikipedia, focusing on temporal vocabulary such as "before," "after," and "during."

## Model
**ELECTRA-small**: A pre-trained language model fine-tuned on the SQuAD dataset.

**Training**: The model was trained on subsets of the SQuAD dataset, enriched with adversarial examples containing temporal vocabulary.

## Evaluation
The model's performance was evaluated on:

The original SQuAD dataset.

Adversarial datasets containing temporal vocabulary.

**Metrics used**: Accuracy and F1 score.

## Results

Training on adversarial examples improved the model's accuracy on temporal QA tasks, but overall accuracy on the SQuAD dataset decreased slightly.

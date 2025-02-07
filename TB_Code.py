import json
import re
from typing import List, Dict, Any
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')


class AdvancedSQuADErrorAnalyzer:
    def __init__(self, predictions_file: str):
        """
        Initializes the analyzer with predictions from a JSONL file.

        Args:
            predictions_file (str): Path to the JSONL file containing model predictions.
        """
        self.predictions = self._load_predictions(predictions_file)
        self.stop_words = set(stopwords.words('english'))  # Load English stopwords

    def _load_predictions(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Loads predictions from a JSONL file.

        Args:
            filepath (str): Path to the JSONL file.

        Returns:
            List[Dict[str, Any]]: List of prediction dictionaries.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def accuracy_by_entity_type(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates accuracy based on the type of entity in the predicted answer.

        Returns:
            Dict[str, Dict[str, float]]: Accuracy statistics for each entity type.
        """
        entity_type_stats = {
            'number': {'total': 0, 'correct': 0},
            'proper_noun': {'total': 0, 'correct': 0},
            'common_noun': {'total': 0, 'correct': 0},
            'other': {'total': 0, 'correct': 0}
        }

        def get_entity_type(text: str) -> str:
            """
            Determines the entity type of a given text.

            Args:
                text (str): The text to classify.

            Returns:
                str: Entity type ('number', 'proper_noun', 'common_noun', or 'other').
            """
            if re.search(r'\d+', text):  # Check if the text contains numbers
                return 'number'
            elif text.istitle():  # Check if the text is a proper noun (title case)
                return 'proper_noun'
            elif text.islower():  # Check if the text is a common noun (lowercase)
                return 'common_noun'
            else:
                return 'other'

        for pred in self.predictions:
            true_answers = pred['answers']['text']
            predicted_answer = pred['predicted_answer']
            pred_entity_type = get_entity_type(predicted_answer)
            entity_type_stats[pred_entity_type]['total'] += 1

            # Check if the predicted answer matches any of the true answers
            if any(predicted_answer.lower().strip() == ans.lower().strip() for ans in true_answers):
                entity_type_stats[pred_entity_type]['correct'] += 1

        # Calculate accuracy for each entity type
        accuracies = {}
        for entity_type, stats in entity_type_stats.items():
            total = stats['total']
            correct = stats['correct']
            accuracies[entity_type] = {
                'total_count': total,
                'accuracy': correct / total if total > 0 else 0,
                'correct_count': correct
            }
        return accuracies

    def accuracy_by_partial_match(self) -> Dict[str, float]:
        """
        Calculates accuracy based on partial matches between predicted and true answers.

        Returns:
            Dict[str, float]: Percentage of exact matches, partial matches, and no matches.
        """
        partial_match_stats = {
            'exact_match': 0,
            'partial_match': 0,
            'no_match': 0
        }
        total_predictions = len(self.predictions)

        for pred in self.predictions:
            true_answers = pred['answers']['text']
            predicted_answer = pred['predicted_answer']

            # Check for exact matches
            if any(predicted_answer.lower().strip() == ans.lower().strip() for ans in true_answers):
                partial_match_stats['exact_match'] += 1
            # Check for partial matches
            elif any(predicted_answer.lower() in ans.lower() or
                     ans.lower() in predicted_answer.lower()
                     for ans in true_answers):
                partial_match_stats['partial_match'] += 1
            else:
                partial_match_stats['no_match'] += 1

        # Convert counts to percentages
        return {
            key: count / total_predictions * 100
            for key, count in partial_match_stats.items()
        }

    def accuracy_by_question_type(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates accuracy based on the type of question (e.g., who, what, where).

        Returns:
            Dict[str, Dict[str, float]]: Accuracy statistics for each question type.
        """

        def categorize_question(question: str) -> str:
            """
            Categorizes the question based on its starting word.

            Args:
                question (str): The question to categorize.

            Returns:
                str: Question type ('who', 'what', 'where', etc.).
            """
            question = question.lower()
            if question.startswith('who'):
                return 'who'
            elif question.startswith('what'):
                return 'what'
            elif question.startswith('where'):
                return 'where'
            elif question.startswith('when'):
                return 'when'
            elif question.startswith('why'):
                return 'why'
            elif question.startswith('how'):
                return 'how'
            else:
                return 'other'

        question_type_counts = {}

        for pred in self.predictions:
            question = pred['question']
            true_answers = pred['answers']['text']
            predicted_answer = pred['predicted_answer']
            q_type = categorize_question(question)

            # Initialize counts for new question types
            if q_type not in question_type_counts:
                question_type_counts[q_type] = {
                    'total': 0,
                    'correct': 0
                }

            question_type_counts[q_type]['total'] += 1

            # Check if the predicted answer matches any of the true answers
            if any(predicted_answer.lower().strip() == ans.lower().strip() for ans in true_answers):
                question_type_counts[q_type]['correct'] += 1

        # Calculate accuracy for each question type
        accuracies = {}
        for q_type, stats in question_type_counts.items():
            total = stats['total']
            correct = stats['correct']
            accuracies[q_type] = {
                'total_count': total,
                'accuracy': correct / total if total > 0 else 0,
                'correct_count': correct
            }
        return accuracies

    def analyze_incorrect_context_words(self, top_n: int = 50) -> Dict[str, int]:
        """
        Analyzes the most common words in contexts where the model made incorrect predictions.

        Args:
            top_n (int): Number of top words to return.

        Returns:
            Dict[str, int]: Dictionary of the most common words and their frequencies.
        """
        # Collect contexts for incorrect predictions
        incorrect_contexts = [
            pred['context'] for pred in self.predictions
            if not any(pred['predicted_answer'].lower().strip() == ans.lower().strip()
                       for ans in pred['answers']['text'])
        ]

        # Tokenize and filter words
        all_words = []
        for context in incorrect_contexts:
            words = word_tokenize(context.lower())
            words = [
                word for word in words
                if word not in self.stop_words
                   and word.isalnum()
                   and len(word) > 2
            ]
            all_words.extend(words)

        # Return the most common words
        return dict(Counter(all_words).most_common(top_n))

    def analyze_when_questions(self, output_dir: str = './when_questions_analysis_to'):
        """
        Analyzes 'when' questions, saving correct and incorrect predictions to files.

        Args:
            output_dir (str): Directory to save analysis results.

        Returns:
            Dict[str, int]: Statistics for 'when' questions.
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        correct_when_questions_path = os.path.join(output_dir, 'correct_when_questions.jsonl')
        incorrect_when_questions_path = os.path.join(output_dir, 'incorrect_when_questions.jsonl')
        when_questions_stats = {
            'total': 0,
            'correct': 0,
            'incorrect': 0
        }

        with open(correct_when_questions_path, 'w') as correct_file, \
                open(incorrect_when_questions_path, 'w') as incorrect_file:

            for pred in self.predictions:
                if not pred['question'].lower().startswith('when'):
                    continue
                when_questions_stats['total'] += 1

                # Check if the prediction is correct
                is_correct = any(
                    pred['predicted_answer'].lower().strip() == ans.lower().strip()
                    for ans in pred['answers']['text']
                )

                question_data = {
                    'question': pred['question'],
                    'context': pred['context'],
                    'gold_answers': pred['answers']['text'],
                    'predicted_answer': pred['predicted_answer']
                }

                if is_correct:
                    when_questions_stats['correct'] += 1
                    correct_file.write(json.dumps(question_data) + '\n')
                else:
                    when_questions_stats['incorrect'] += 1
                    incorrect_file.write(json.dumps(question_data) + '\n')

        print("\nWHEN Questions Analysis:")
        print(json.dumps(when_questions_stats, indent=2))

        return when_questions_stats


def main(predictions_file: str):
    """
    Main function to run the error analysis.

    Args:
        predictions_file (str): Path to the JSONL file containing model predictions.
    """
    analyzer = AdvancedSQuADErrorAnalyzer(predictions_file)

    print("Accuracy by Entity Type:")
    entity_type_accuracy = analyzer.accuracy_by_entity_type()
    print(json.dumps(entity_type_accuracy, indent=2))

    print("\nPartial Match Analysis:")
    partial_match_accuracy = analyzer.accuracy_by_partial_match()
    print(json.dumps(partial_match_accuracy, indent=2))

    print("\nAccuracy by Question Type:")
    question_type_accuracy = analyzer.accuracy_by_question_type()
    print(json.dumps(question_type_accuracy, indent=2))

    print("\nMost Common Words in Incorrect Contexts:")
    incorrect_context_words = analyzer.analyze_incorrect_context_words()
    print(json.dumps(incorrect_context_words, indent=2))

    print("\nWHEN Questions Detailed Analysis:")
    when_questions_analysis = analyzer.analyze_when_questions()


if __name__ == "__main__":
    main('eval_to/eval_predictions.jsonl')
"""Synthetic data generation for RAG evaluation."""

from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from ..config import settings
from ..document_processing.chunking import Chunk


class SyntheticDataGenerator:
    """
    Generate synthetic question-answer pairs from chunks.

    Following Jason Liu's methodology: generate 5 questions per chunk,
    targeting 97% recall on synthetic data before moving to real users.
    """

    DEFAULT_QUESTION_PROMPT = """Generate {num_questions} diverse questions that can be answered using the following text chunk.

Text Chunk:
{chunk_text}

Requirements:
- Questions should be natural and realistic
- Vary the question types: factual, analytical, comparative, etc.
- Questions should be answerable using ONLY this chunk
- Make questions specific and focused
- Include both simple and complex questions

Generate exactly {num_questions} questions, one per line.
"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        questions_per_chunk: Optional[int] = None,
    ):
        """
        Initialize synthetic data generator.

        Args:
            model: Model to use for generation
            api_key: OpenAI API key
            questions_per_chunk: Number of questions to generate per chunk
        """
        self.model = model
        self.api_key = api_key or settings.openai_api_key
        self.questions_per_chunk = (
            questions_per_chunk or settings.synthetic_questions_per_chunk
        )

        if not self.api_key:
            logger.warning("No OpenAI API key provided. Synthetic data generation will fail.")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized SyntheticDataGenerator with model: {self.model}")

    def generate_questions(
        self,
        chunk: Chunk,
        num_questions: Optional[int] = None,
    ) -> List[str]:
        """
        Generate questions for a single chunk.

        Args:
            chunk: Text chunk
            num_questions: Number of questions to generate

        Returns:
            List of generated questions
        """
        num_questions = num_questions or self.questions_per_chunk

        # Format prompt
        prompt = self.DEFAULT_QUESTION_PROMPT.format(
            num_questions=num_questions,
            chunk_text=chunk.text
        )

        try:
            # Generate questions
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating evaluation questions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500,
            )

            # Parse questions (one per line)
            questions_text = response.choices[0].message.content
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]

            # Remove numbering if present
            questions = [
                q.split('. ', 1)[1] if '. ' in q and q[0].isdigit() else q
                for q in questions
            ]

            logger.debug(f"Generated {len(questions)} questions for chunk {chunk.chunk_id}")

            return questions[:num_questions]

        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []

    def generate_dataset(
        self,
        chunks: List[Chunk],
        num_questions_per_chunk: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic evaluation dataset from chunks.

        Args:
            chunks: List of chunks
            num_questions_per_chunk: Questions per chunk

        Returns:
            List of dictionaries with 'question', 'chunk_id', 'expected_text'
        """
        num_questions = num_questions_per_chunk or self.questions_per_chunk

        logger.info(f"Generating synthetic dataset from {len(chunks)} chunks")

        dataset = []

        for chunk in chunks:
            questions = self.generate_questions(chunk, num_questions)

            for question in questions:
                dataset.append({
                    "question": question,
                    "chunk_id": chunk.chunk_id,
                    "expected_text": chunk.text,
                    "metadata": chunk.metadata,
                })

        logger.info(f"Generated {len(dataset)} question-answer pairs")

        return dataset

    def generate_with_answers(
        self,
        chunk: Chunk,
        num_questions: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate questions with expected answers.

        Args:
            chunk: Text chunk
            num_questions: Number of questions to generate

        Returns:
            List of dicts with 'question' and 'answer'
        """
        num_questions = num_questions or self.questions_per_chunk

        prompt = f"""Generate {num_questions} question-answer pairs based on this text:

Text:
{chunk.text}

For each pair:
Q: [question]
A: [answer based on the text]

Generate exactly {num_questions} pairs.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800,
            )

            content = response.choices[0].message.content

            # Parse Q&A pairs
            pairs = []
            lines = content.split('\n')
            current_q = None

            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    current_q = line[2:].strip()
                elif line.startswith('A:') and current_q:
                    answer = line[2:].strip()
                    pairs.append({
                        "question": current_q,
                        "answer": answer,
                        "chunk_id": chunk.chunk_id
                    })
                    current_q = None

            return pairs[:num_questions]

        except Exception as e:
            logger.error(f"Error generating Q&A pairs: {e}")
            return []

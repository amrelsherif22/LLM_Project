import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent/"tests.jsonl")

class TestQuestions(BaseModel):

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def get_tests():
    tests = []
    with open(TEST_FILE, "r", encoding="UTF-8") as f:
        for line in f:
            data = json.loads(line)
            tests.append(TestQuestions(**data))
    return tests
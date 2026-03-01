from pydantic import BaseModel, Field
import dotenv
import json
from pathlib import Path

TEST_FILE = str(Path(__file__).parent/"test.jsonl")
class TestQuestion(BaseModel):
    question : str=Field(description="a question the user will provide to the agent")
    keywords : list(str)=Field(description="words that should be looked up against")
    reference_answer : str=Field(description="an answer to the question")
    category:str=Field(description="Category of where it belongs to")


def load_tests():
    test = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            test.append(TestQuestion(**data))
    return  test
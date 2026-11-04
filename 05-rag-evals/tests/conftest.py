import pytest

from rag_eval.generate import build_documents, build_questions


@pytest.fixture(scope="session")
def docs():
    return build_documents()


@pytest.fixture(scope="session")
def questions():
    return build_questions()

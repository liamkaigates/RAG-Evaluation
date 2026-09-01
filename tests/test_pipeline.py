import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_eval.data import load_documents, load_questions
from rag_eval.evaluate import evaluate_questions
from rag_eval.generator import generate_answer
from rag_eval.ingest import ingest_directory, write_corpus
from rag_eval.retriever import build_index, load_index, save_index


class RagPipelineTest(unittest.TestCase):
    def test_search_and_generate(self):
        with TemporaryDirectory() as tmp:
            index = persist_index(load_documents("data/corpus.jsonl"), Path(tmp) / "index")
            results = index.search("Which metrics evaluate retrieval quality?", top_k=3)
            generated = generate_answer("Which metrics evaluate retrieval quality?", results, provider="local")

            self.assertGreater(len(results), 0)
            self.assertIn("evaluation", {result.id for result in results})
            self.assertTrue(hasattr(results[0], "sparse_score"))
            self.assertTrue(hasattr(results[0], "dense_score"))
            self.assertTrue(generated["answer"])
            self.assertIn("[S", generated["answer"])
            self.assertTrue(generated["citations"])
            self.assertTrue(generated["sources"])

    def test_evaluation_metrics(self):
        with TemporaryDirectory() as tmp:
            index = persist_index(load_documents("data/corpus.jsonl"), Path(tmp) / "index")
            questions = load_questions("data/eval_questions.jsonl")
            payload = evaluate_questions(index, questions, top_k=4, generation_provider="local")

            self.assertEqual(payload["summary"]["question_count"], float(len(questions)))
            self.assertGreaterEqual(payload["summary"]["recall_at_k"], 0.8)
            self.assertGreaterEqual(payload["summary"]["mrr"], 0.5)
            self.assertGreater(len(payload["questions"]), 0)

    def test_ingests_internal_document_chunks(self):
        with TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "docs"
            source_dir.mkdir()
            (source_dir / "handbook.md").write_text(
                "# Handbook\n\n## Access\n\nEmployees use source citations for every generated answer. "
                "Audit logs store retrieved chunks and answer citations.",
                encoding="utf-8",
            )
            corpus = Path(tmp) / "corpus.jsonl"
            documents = ingest_directory(source_dir, max_words=20, overlap=4)
            write_corpus(documents, corpus)
            index = persist_index(load_documents(corpus), Path(tmp) / "index")
            results = index.search("What do audit logs store?", top_k=2)

            self.assertGreaterEqual(len(documents), 1)
            self.assertIn("handbook.md", documents[0].source_path)
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("Audit", results[0].text)

    def test_loads_legacy_pickle_metadata(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "index"
            index = persist_index(load_documents("data/corpus.jsonl"), path)
            legacy = {
                "documents": index.documents,
                "embedding_provider": index.embedding_model.name,
                "embedding_dimension": index.embedding_model.dimension,
                "sparse_weight": index.sparse_weight,
            }
            (path / "metadata.pkl").write_bytes(pickle.dumps(legacy))
            (path / "metadata.json").unlink()

            loaded = load_index(path, embedding_provider="local")
            results = loaded.search("Which metrics evaluate retrieval quality?", top_k=3)

            self.assertEqual(len(loaded.documents), len(index.documents))
            self.assertGreater(len(results), 0)


def persist_index(documents, path: Path):
    index = build_index(documents, embedding_provider="local")
    save_index(index, path)
    return load_index(path, embedding_provider="local")


if __name__ == "__main__":
    unittest.main()

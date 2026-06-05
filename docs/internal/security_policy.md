# Security Policy

## Data Access

Internal documentation ingestion should run with least-privilege access. Connectors should read only approved folders and should record the source path for every indexed chunk. Sensitive documents can be filtered before indexing.

## Auditability

Production RAG systems should log the question, retrieved chunk identifiers, answer citations, latency, and index version. Audit logs help teams investigate unsupported answers, stale documentation, and retrieval failures.

## Deployment

The RAG API should run behind authentication and rate limiting. Index artifacts should be versioned so teams can roll back to a prior documentation snapshot when a new ingest job degrades answer quality.

# RAG Platform Architecture

## Overview

The internal RAG platform answers employee questions by retrieving approved company documentation and generating answers only from that retrieved context. The service stores every indexed passage with file path, section, and chunk metadata so answers can cite their sources.

## Retrieval

The retriever uses hybrid search. Sparse keyword search finds exact terminology, product names, policy codes, and acronyms. Dense vector search finds semantically related passages when employees ask questions using different wording than the documentation. Scores are normalized and fused before ranking.

## Grounding

Generated answers must include inline source citations. If retrieved context is weak or unsupported, the service should return a low-confidence answer or decline to answer instead of inventing facts.

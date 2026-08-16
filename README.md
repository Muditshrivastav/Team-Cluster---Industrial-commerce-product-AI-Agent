# Team-Cluster — Industrial Commerce Product AI Agent

## Hybrid RAG (Supabase)

`hybrid_rag/` replaces the SQLite + Chroma split with one Supabase/Postgres
store. Exact normalized manufacturer part-number lookup wins; otherwise,
Postgres full-text and pgvector cosine candidates are fused with reciprocal-rank
fusion. Only validated products are written to the retrieval index.

1. Copy `.env.example` to `.env` and add the project URL and **server-only**
   service-role key. Do not expose that key to Streamlit client code.
2. Run `supabase/schema.sql` in the project SQL Editor. It creates the table,
   pgvector index, hybrid search RPC, RLS, and least-privilege grants.
3. In **Database → Data API**, expose `rag_products` and
   `hybrid_rag_search_products` to the Data API for the `service_role` only. New
   Supabase projects no longer expose public-schema entities automatically.
4. Install the pinned dependencies: `pip install -r requirements.txt`.
5. Generate 1,536-dimensional embeddings with the embedding provider selected
   by the backend and pass them to the repository.

See `hybrid_rag/example_usage.py` for `save_validated_product()` and
`retrieve_similar_products()`.

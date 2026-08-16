-- Apply in Supabase SQL Editor before running the backend.
create extension if not exists vector with schema extensions;
create extension if not exists pgcrypto with schema extensions;

-- Dedicated table: do not reuse the team's existing public.products table.
create table if not exists public.rag_products (
  id uuid primary key default extensions.gen_random_uuid(),
  manufacturer_part_number text not null,
  normalized_mpn text generated always as (upper(regexp_replace(manufacturer_part_number, '[^A-Za-z0-9]', '', 'g'))) stored,
  short_description text,
  specifications jsonb not null default '{}'::jsonb,
  source_urls text[] not null default '{}',
  validation_status text not null default 'pending' check (validation_status in ('pending', 'validated', 'rejected')),
  embedding extensions.vector(1536),
  search_document tsvector generated always as (
    setweight(to_tsvector('simple', coalesce(manufacturer_part_number, '')), 'A') ||
    setweight(to_tsvector('simple', coalesce(brand, '')), 'B') ||
    setweight(to_tsvector('simple', coalesce(short_description, '')), 'C') ||
    setweight(to_tsvector('simple', coalesce(specifications::text, '')), 'D')
  ) stored,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (normalized_mpn)
);

create index if not exists rag_products_search_document_idx on public.rag_products using gin (search_document);
create index if not exists rag_products_embedding_hnsw_idx on public.rag_products using hnsw (embedding vector_cosine_ops) where embedding is not null;

create or replace function public.set_rag_products_updated_at()
returns trigger language plpgsql set search_path = '' as $$
begin new.updated_at = now(); return new; end;
$$;

drop trigger if exists rag_products_set_updated_at on public.rag_products;
create trigger rag_products_set_updated_at before update on public.rag_products
for each row execute function public.set_rag_products_updated_at();

-- Exact MPN results always win. Otherwise full-text and vector candidates are
-- fused using reciprocal-rank fusion.
create or replace function public.hybrid_rag_search_products(
  query_mpn text, query_text text, query_embedding extensions.vector(1536), match_count integer default 5
)
returns table (
  id uuid, manufacturer_part_number text, brand text, short_description text,
  specifications jsonb, source_urls text[], validation_status text,
  score double precision, match_strategy text
)
language sql stable set search_path = '' as $$
  with exact_matches as (
    select p.*, 1.0::double precision as final_score from public.rag_products p
    where p.normalized_mpn = upper(regexp_replace(coalesce(query_mpn, ''), '[^A-Za-z0-9]', '', 'g'))
      and coalesce(query_mpn, '') <> '' limit match_count
  ), text_candidates as (
    select p.id, row_number() over (order by ts_rank_cd(p.search_document, websearch_to_tsquery('simple', query_text)) desc) as rank
    from public.rag_products p where p.search_document @@ websearch_to_tsquery('simple', query_text)
    order by ts_rank_cd(p.search_document, websearch_to_tsquery('simple', query_text)) desc limit greatest(match_count * 10, 20)
  ), vector_candidates as (
    select p.id, row_number() over (order by p.embedding OPERATOR(extensions.<=>) query_embedding) as rank
    from public.rag_products p where p.embedding is not null and query_embedding is not null
    order by p.embedding OPERATOR(extensions.<=>) query_embedding limit greatest(match_count * 10, 20)
  ), fused as (
    select coalesce(t.id, v.id) as id,
      coalesce(1.0 / (60 + t.rank), 0.0) + coalesce(1.0 / (60 + v.rank), 0.0) as final_score,
      case when t.id is not null and v.id is not null then 'hybrid' when t.id is not null then 'sparse' else 'dense' end as strategy
    from text_candidates t full outer join vector_candidates v on v.id = t.id
  )
  select e.id, e.manufacturer_part_number, e.brand, e.short_description, e.specifications, e.source_urls, e.validation_status,
         e.final_score as score, 'exact_mpn'::text as match_strategy
  from exact_matches e
  union all
  select p.id, p.manufacturer_part_number, p.brand, p.short_description, p.specifications, p.source_urls, p.validation_status,
         f.final_score as score, f.strategy as match_strategy
  from fused f join public.rag_products p on p.id = f.id where not exists (select 1 from exact_matches)
  order by score desc limit match_count;
$$;

alter table public.rag_products enable row level security;
revoke all on table public.rag_products from anon, authenticated;
revoke all on function public.hybrid_rag_search_products(text, text, extensions.vector, integer) from public, anon, authenticated;
grant all on table public.rag_products to service_role;
grant execute on function public.hybrid_rag_search_products(text, text, extensions.vector, integer) to service_role;

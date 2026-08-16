"""Server-side repository for exact, sparse, and semantic product retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence
from uuid import UUID

from dotenv import load_dotenv
from supabase import Client, create_client

EMBEDDING_DIMENSIONS = 1536


@dataclass(frozen=True)
class ProductRecord:
    manufacturer_part_number: str
    brand: str
    short_description: str | None = None
    specifications: dict[str, Any] = field(default_factory=dict)
    source_urls: list[str] = field(default_factory=list)
    validation_status: str = "pending"
    embedding: Sequence[float] | None = None

    def to_row(self) -> dict[str, Any]:
        if self.embedding is not None and len(self.embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Expected a {EMBEDDING_DIMENSIONS}-dimension embedding.")
        return {
            "manufacturer_part_number": self.manufacturer_part_number,
            "brand": self.brand,
            "short_description": self.short_description,
            "specifications": self.specifications,
            "source_urls": self.source_urls,
            "validation_status": self.validation_status,
            "embedding": list(self.embedding) if self.embedding is not None else None,
        }


@dataclass(frozen=True)
class HybridSearchResult:
    id: UUID
    manufacturer_part_number: str
    brand: str
    short_description: str | None
    specifications: dict[str, Any]
    source_urls: list[str]
    validation_status: str
    score: float
    match_strategy: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "HybridSearchResult":
        return cls(
            id=UUID(row["id"]), manufacturer_part_number=row["manufacturer_part_number"],
            brand=row["brand"], short_description=row.get("short_description"),
            specifications=row["specifications"], source_urls=row["source_urls"],
            validation_status=row["validation_status"], score=float(row["score"]),
            match_strategy=row["match_strategy"],
        )


class HybridProductRepository:
    """Trusted-backend access only; do not instantiate this in browser code."""

    def __init__(self, client: Client | None = None) -> None:
        self._client = client or self._create_server_client()

    @staticmethod
    def _create_server_client() -> Client:
        load_dotenv()
        url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured.")
        return create_client(url, key)

    def save_validated_product(self, product: ProductRecord) -> ProductRecord:
        """Upsert a validated product so it becomes available for retrieval."""
        if product.validation_status != "validated":
            raise ValueError("Only validated products may be indexed for retrieval.")
        self._client.table("rag_products").upsert(product.to_row(), on_conflict="normalized_mpn").execute()
        return product

    def retrieve_similar_products(
        self, *, query_text: str, query_embedding: Sequence[float] | None,
        manufacturer_part_number: str | None = None, limit: int = 5,
    ) -> list[HybridSearchResult]:
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20.")
        if query_embedding is not None and len(query_embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Expected a {EMBEDDING_DIMENSIONS}-dimension embedding.")
        response = self._client.rpc("hybrid_rag_search_products", {
            "query_mpn": manufacturer_part_number or "", "query_text": query_text,
            "query_embedding": list(query_embedding) if query_embedding is not None else None,
            "match_count": limit,
        }).execute()
        return [HybridSearchResult.from_row(row) for row in response.data]

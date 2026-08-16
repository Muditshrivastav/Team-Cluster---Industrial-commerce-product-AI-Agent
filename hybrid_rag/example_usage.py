"""Replace the embedding argument with the application's embedding-provider result."""
from hybrid_rag import HybridProductRepository, ProductRecord


def index_validated_product(embedding: list[float]) -> None:
    HybridProductRepository().save_validated_product(ProductRecord(
        manufacturer_part_number="ABC-123", brand="Example Manufacturer",
        short_description="Industrial temperature sensor with M12 connector",
        specifications={"connector": "M12", "measurement_range": "-40 to 125 C"},
        source_urls=["https://example.com/abc-123"], validation_status="validated",
        embedding=embedding,
    ))


def retrieve(query_embedding: list[float]) -> None:
    results = HybridProductRepository().retrieve_similar_products(
        manufacturer_part_number="ABC-123",
        query_text="Example Manufacturer industrial temperature sensor M12",
        query_embedding=query_embedding,
    )
    for result in results:
        print(result.match_strategy, result.score, result.manufacturer_part_number)

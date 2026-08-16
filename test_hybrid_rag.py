"""Smoke test for the Supabase-backed hybrid RAG repository."""

from hybrid_rag import HybridProductRepository, ProductRecord


DIMENSIONS = 1536
SENSOR_VECTOR = [1.0] + [0.0] * (DIMENSIONS - 1)
VALVE_VECTOR = [0.0, 1.0] + [0.0] * (DIMENSIONS - 2)


def main() -> None:
    repo = HybridProductRepository()

    print("1/3 Saving test products...")
    repo.save_validated_product(ProductRecord(
        manufacturer_part_number="TEST-RAG-SENSOR-001",
        brand="TestCo",
        short_description="Industrial temperature sensor",
        validation_status="validated",
        embedding=SENSOR_VECTOR,
    ))
    repo.save_validated_product(ProductRecord(
        manufacturer_part_number="TEST-RAG-VALVE-001",
        brand="TestCo",
        short_description="Industrial control valve",
        validation_status="validated",
        embedding=VALVE_VECTOR,
    ))

    print("2/3 Testing exact MPN retrieval...")
    exact = repo.retrieve_similar_products(
        manufacturer_part_number="TEST-RAG-SENSOR-001",
        query_text="industrial temperature sensor",
        query_embedding=SENSOR_VECTOR,
    )
    assert exact and exact[0].match_strategy == "exact_mpn", exact
    assert exact[0].manufacturer_part_number == "TEST-RAG-SENSOR-001", exact

    print("3/3 Testing semantic retrieval...")
    semantic = repo.retrieve_similar_products(
        query_text="industrial temperature sensor",
        query_embedding=SENSOR_VECTOR,
    )
    assert semantic and semantic[0].manufacturer_part_number == "TEST-RAG-SENSOR-001", semantic

    print("PASS: hybrid RAG is connected and working.")


if __name__ == "__main__":
    main()

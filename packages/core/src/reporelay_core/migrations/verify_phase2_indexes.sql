-- Phase 2 Index Verification: EXPLAIN ANALYZE on top queries from data design doc
-- These queries correspond to the data design doc (compass/10-data-collection-design.md)
-- Even with no data, EXPLAIN reveals whether indexes are selected by the planner.

-- Query 1: Semantic neighbors (ANN via pgvector HNSW)
-- Demonstrates: ix_readme_texts_embedding_hnsw
-- Uses EXPLAIN (no ANALYZE) to show plan without executing vector ops on empty table.
EXPLAIN
SELECT repo_id, 1 - (embedding <=> array_fill(0::real, ARRAY[768])::vector) AS similarity
FROM readme_texts
ORDER BY embedding <=> array_fill(0::real, ARRAY[768])::vector
LIMIT 200;

-- Query 2: Two-hop neighbor lookup (graph proximity via precomputed table)
-- Demonstrates: ix_two_hop_source
EXPLAIN ANALYZE
SELECT neighbor_repo_id, combined_weight
FROM two_hop_neighbors
WHERE source_repo_id = 1
ORDER BY combined_weight DESC
LIMIT 200;

-- Query 3: Co-starred repos (item-based collaborative filtering)
-- Demonstrates: ix_star_events_user_repo, ix_star_events_repo_time
EXPLAIN ANALYZE
SELECT s2.repo_id, COUNT(*) AS co_star_count
FROM star_events s1
JOIN star_events s2 ON s1.user_id = s2.user_id AND s1.repo_id != s2.repo_id
WHERE s1.repo_id = 1
GROUP BY s2.repo_id
ORDER BY co_star_count DESC
LIMIT 200;

-- Query 4: Contributor overlap (ecosystem neighbors)
-- Demonstrates: ix_contributor_edges_repo
EXPLAIN ANALYZE
SELECT c2.repo_id AS repo_b, COUNT(DISTINCT c1.user_id) AS shared_contributors
FROM contributor_edges c1
JOIN contributor_edges c2 ON c1.user_id = c2.user_id AND c1.repo_id != c2.repo_id
WHERE c1.repo_id = 1
GROUP BY c2.repo_id
ORDER BY shared_contributors DESC
LIMIT 200;

-- Query 5: Dependency lookup (direct dependency neighbors)
-- Demonstrates: ix_dep_edges_repo
EXPLAIN ANALYZE
SELECT dependency_name, ecosystem, version_constraint
FROM dependency_edges
WHERE repo_id = 1;

-- Bonus: Verify all Phase 2 indexes are present
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND indexname IN (
    'ix_repos_language',
    'ix_repos_topics',
    'ix_repos_stars',
    'ix_star_events_user_repo',
    'ix_star_events_repo_time',
    'ix_star_events_recent',
    'ix_contributor_edges_repo',
    'ix_dep_edges_repo',
    'ix_two_hop_source',
    'ix_readme_texts_embedding_hnsw'
  )
ORDER BY indexname;

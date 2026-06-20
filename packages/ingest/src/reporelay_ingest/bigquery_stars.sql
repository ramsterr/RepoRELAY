-- GitHub Archive BigQuery SQL for extracting star events
-- Execute in Google BigQuery on the `githubarchive` public dataset

-- Schema reference (WatchEvent = star):
--   type: STRING       - 'WatchEvent' for stars
--   repo.name: STRING  - 'owner/repo' format
--   actor.login: STRING - user who starred
--   actor.id: INTEGER  - user id
--   repo.id: INTEGER   - repository id
--   created_at: TIMESTAMP

-- Query: Extract star events for our seed repos from the last 6 months
SELECT
    actor.id AS user_id,
    actor.login AS user_login,
    repo.id AS repo_id,
    repo.name AS repo_full_name,
    created_at AS starred_at
FROM `githubarchive.month.*`
WHERE
    _TABLE_SUFFIX BETWEEN '202601' AND '202606'
    AND type = 'WatchEvent'
    AND repo.name IN (
        -- Replace with actual repo full_names from your seed dataset
        'vercel/next.js',
        'freeCodeCamp/freeCodeCamp',
        'react/react'
    )
ORDER BY starred_at DESC

-- Alternative: query against a list of repo IDs
-- WHERE repo.id IN (70107786, 28457823, 10270250)

-- Export as CSV/JSON and use `ingest load-stars` to bulk import

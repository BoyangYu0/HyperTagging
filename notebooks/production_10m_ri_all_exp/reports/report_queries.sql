-- summary
SELECT validated_events, unique_event_uids, completed_shards,
       ROUND(output_gib, 3) AS output_gib,
       ROUND(bytes_per_event, 1) AS bytes_per_event,
       klm_nodes, missing_shards, non_whitespace_stderr,
       input_files, experiments
FROM campaign_summary;

-- category
SELECT category, COUNT(*) AS shards, SUM(events) AS events,
       ROUND(SUM(output_mib) / 1024.0, 3) AS output_gib,
       ROUND(AVG(events_per_second), 2) AS mean_events_per_second,
       ROUND(MAX(peak_rss_mib), 1) AS max_peak_rss_mib,
       SUM(klm_nodes) AS klm_nodes
FROM shard_metrics
GROUP BY category
ORDER BY events DESC, category;

-- resources
SELECT task_id, category, events_per_second,
       peak_rss_mib, 8192 AS requested_memory_mib,
       elapsed_seconds, validation_seconds, output_mib
FROM shard_metrics
ORDER BY task_id;

-- leaf_modes
SELECT label AS leaf_mode, CAST(value AS INTEGER) AS nodes
FROM topology_summary
WHERE family = 'leaf_mode'
ORDER BY nodes DESC;

-- topology_quantiles
SELECT family, label AS quantile, value
FROM topology_summary
WHERE family IN ('node_count_quantile', 'depth_quantile')
ORDER BY family, value;

-- klm
SELECT category, SUM(klm_nodes) AS klm_nodes
FROM shard_metrics
GROUP BY category
ORDER BY klm_nodes DESC, category;

-- provenance
SELECT property, value
FROM provenance_summary
ORDER BY property;

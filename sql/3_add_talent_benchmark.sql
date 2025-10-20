INSERT INTO talent_benchmarks (
  role_name, job_level, role_purpose, selected_talent_ids, weights_config
)
VALUES (
  'Brand Executive',
  'V',
  'Analyze data for business insight',
  ARRAY['100605', '101493'],
  '{
    "Leadership": 0.2,
    "Cognitive": 0.05,
    "Followership": 0.15,
    "Social": 0.2,
    "Discipline": 0.15,
    "Decision": 0.25
  }'::jsonb
);

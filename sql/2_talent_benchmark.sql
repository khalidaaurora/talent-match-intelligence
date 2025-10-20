CREATE TABLE IF NOT EXISTS talent_benchmarks (
  job_vacancy_id serial PRIMARY KEY,
  role_name text,
  job_level text,
  role_purpose text,
  selected_talent_ids text[],
  weights_config jsonb,
  created_at timestamp DEFAULT now()
);

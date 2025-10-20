-- Step 1: Select all high performers (rating = 5) as the benchmark group
WITH selected_benchmarks AS (
  SELECT employee_id
  FROM employee
  WHERE is_high = '1.0'
),

-- Step 2: Calculate the median score (baseline) for each Talent Variable (TV)
-- grouped by its Talent Group Variable (TGV)
benchmark_scores AS (
  SELECT
    m.tgv_name,
    es.tv_name,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY es.score) AS baseline_score
  FROM employee_score es
  JOIN selected_benchmarks sb ON sb.employee_id = es.employee_id
  JOIN tv_meta m ON es.tv_name = m.tv_name
  GROUP BY m.tgv_name, es.tv_name
),

-- Step 3: Compare every employee’s score with the benchmark median
-- to calculate each TV’s match rate (how close they are to benchmark)
tv_match AS (
  SELECT
    e.employee_id,
    e.directorate_name AS directorate,
    e.position_name AS role,
    e.grade_name AS grade,
    m.tgv_name,
    es.tv_name,
    b.baseline_score,
    es.score AS user_score,
    CASE 
      WHEN es.score >= b.baseline_score THEN 100
      ELSE ROUND((es.score / b.baseline_score * 100)::numeric, 2)
    END AS tv_match_rate
  FROM employee_score es
  JOIN employee e ON e.employee_id = es.employee_id
  JOIN tv_meta m ON es.tv_name = m.tv_name
  JOIN benchmark_scores b ON b.tv_name = es.tv_name
),

-- Step 4: Aggregate the TV match rates within each TGV
-- to calculate average match score per TGV per employee
tgv_match AS (
  SELECT
    employee_id,
    tgv_name,
    ROUND(AVG(tv_match_rate)::numeric, 2) AS tgv_match_rate
  FROM tv_match
  GROUP BY employee_id, tgv_name
),

-- Step 5: Combine all TGVs into a final overall match rate per employee
final_match AS (
  SELECT
    employee_id,
    ROUND(AVG(tgv_match_rate)::numeric, 2) AS final_match_rate
  FROM tgv_match
  GROUP BY employee_id
)

-- Step 6: Final output
-- Show all key columns: candidate info, TV-level comparison, TGV averages,
-- and their final overall match rate, sorted by the best fit first
SELECT
  tm.employee_id AS candidate_id,
  tm.directorate,
  tm.role,
  tm.grade,
  tm.tgv_name,
  tm.tv_name,
  tm.baseline_score,
  tm.user_score,
  tm.tv_match_rate,
  tg.tgv_match_rate,
  fm.final_match_rate
FROM tv_match tm
JOIN tgv_match tg 
  ON tm.employee_id = tg.employee_id 
  AND tm.tgv_name = tg.tgv_name
JOIN final_match fm 
  ON tm.employee_id = fm.employee_id
ORDER BY fm.final_match_rate DESC, tm.employee_id, tm.tgv_name, tm.tv_name;
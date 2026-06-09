-- Custom singular data governance test: Exposure must be strictly greater than 0.
-- Reinsurance portfolios require active coverage duration. Exposure <= 0 is invalid.
select
    policy_id,
    exposure
from {{ ref('stg_policies') }}
where exposure <= 0

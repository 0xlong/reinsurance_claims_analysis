-- Custom singular data governance test: Claim count cannot be negative.
-- Underwriters require non-negative count values.
select
    policy_id,
    claim_count
from {{ ref('stg_policies') }}
where claim_count < 0

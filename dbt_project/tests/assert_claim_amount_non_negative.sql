-- Custom singular data governance test: Total claim amount cannot be negative.
-- Reinsurance claims cannot represent negative payouts (except for salvage/subrogation, but raw intake should not be negative).
select
    policy_id,
    total_claim_amount
from {{ ref('stg_claims') }}
where total_claim_amount < 0

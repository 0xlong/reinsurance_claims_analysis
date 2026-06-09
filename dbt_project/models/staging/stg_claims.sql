{{ config(materialized='view') }}

-- Staging model for claim severity data.
-- Aggregates claim amounts by policy ID since severity data has one row per claim, 
-- but we need policy-level aggregation for the ML pricing model.
with source as (
    select * from {{ source('raw_source', 'raw_claims') }}
),

aggregated as (
    select
        cast(idpol as bigint) as policy_id,
        sum(cast(claimamount as double)) as total_claim_amount,
        count(*) as claim_event_count
    from source
    group by 1
)

select * from aggregated

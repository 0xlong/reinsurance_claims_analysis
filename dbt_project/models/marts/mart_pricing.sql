{{ config(materialized='table') }}

-- Mart model: mart_pricing
-- Joins policy frequency data with aggregated claims severity data and regions lookup metadata
-- to produce the final, clean, ML-ready dataset for modeling pure premium.
with policies as (
    select * from {{ ref('stg_policies') }}
),

claims as (
    select * from {{ ref('stg_claims') }}
),

regions as (
    select * from {{ ref('stg_regions') }}
)

select
    -- Dimensions / Identifiers
    p.policy_id,
    
    -- Exposure (Sample Weight)
    p.exposure,
    
    -- Risk features
    p.area_code,
    p.vehicle_power,
    p.vehicle_age,
    p.driver_age,
    p.bonus_malus,
    p.vehicle_brand,
    p.fuel_type,
    p.population_density,
    p.region_code,
    r.region_name,
    r.region_population,
    
    -- Targets for ML Modeling
    p.claim_count,
    coalesce(c.total_claim_amount, 0.0) as total_claim_amount,
    
    -- Derived columns
    case 
        when p.claim_count > 0 then true 
        else false 
    end as has_claim,
    
    case 
        when p.claim_count > 0 then coalesce(c.total_claim_amount, 0.0) / p.claim_count
        else 0.0
    end as avg_claim_amount

from policies p
left join claims c 
    on p.policy_id = c.policy_id
left join regions r
    on p.region_code = r.region_code

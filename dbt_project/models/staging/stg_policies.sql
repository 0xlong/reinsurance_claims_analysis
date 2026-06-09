{{ config(materialized='view') }}

-- Staging model for policy frequency and exposure data.
-- Standardizes column names and casts them to appropriate data types.
with source as (
    select * from {{ source('raw_source', 'raw_policies') }}
)

select
    -- Primary key: Policy ID
    cast(idpol as bigint) as policy_id,
    
    -- Target variable for frequency: Number of claims
    cast(claimnb as integer) as claim_count,
    
    -- Exposure (sample weight): portion of a year the policy was active
    cast(exposure as double) as exposure,
    
    -- Risk factors / Features
    cast(area as varchar) as area_code,
    cast(vehpower as integer) as vehicle_power,
    cast(vehage as integer) as vehicle_age,
    cast(drivage as integer) as driver_age,
    cast(bonusmalus as integer) as bonus_malus,
    cast(vehbrand as varchar) as vehicle_brand,
    cast(vehgas as varchar) as fuel_type,
    cast(density as integer) as population_density,
    cast(region as varchar) as region_code

from source

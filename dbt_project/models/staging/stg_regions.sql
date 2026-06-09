{{ config(materialized='view') }}

-- Staging model for regions metadata.
-- Cleans and casts regions lookup data containing region codes, friendly names, and populations.
with source as (
    select * from {{ source('raw_source', 'raw_regions') }}
)

select
    -- Standardize names and cast types
    -- In regions.csv, headers are Name, Region, Volume
    cast(region as varchar) as region_code,
    cast(name as varchar) as region_name,
    cast(volume as bigint) as region_population
from source

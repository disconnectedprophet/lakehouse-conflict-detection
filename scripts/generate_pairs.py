"""
Generates pairs.json, ddl.sql, lineage.json, and manifests/ for the
datalake conflict-detection dataset.

Target label distribution (proportional to original):
  TYPE1_MEASURE ~32%
  TYPE2_GRANULARITY ~21%
  NO_CONFLICT_DUPLICATE ~30%
  NO_CONFLICT_DIFF_ENTITY ~17%

Output written to ../dataset/:
  pairs.json
  metadata/ddl.sql
  metadata/lineage.json
  metadata/manifests/*.json

Run via:
  python3 generate_pairs.py
"""

import json
import os
import random
import re
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, "..", "dataset", "tables")
META_DIR = os.path.join(SCRIPT_DIR, "..", "dataset", "metadata")
MANIFESTS_DIR = os.path.join(META_DIR, "manifests")

LABELS = ["TYPE1_MEASURE", "TYPE2_GRANULARITY", "NO_CONFLICT_DUPLICATE", "NO_CONFLICT_DIFF_ENTITY"]

PAIR_TEMPLATES: list[dict] = [
    # --- TYPE1_MEASURE: same concept, different unit or encoding ---
    # sales: currency variants (annual_sales column)
    {"a": ("products_sales_usd", "annual_sales_usd"), "b": ("products_sales_eur", "annual_sales_eur"), "label": "TYPE1_MEASURE"},
    {"a": ("products_sales_usd", "annual_sales_usd"), "b": ("products_sales_gbp", "annual_sales_gbp"), "label": "TYPE1_MEASURE"},
    {"a": ("products_sales_usd", "annual_sales_usd"), "b": ("products_sales_jpy", "annual_sales_jpy"), "label": "TYPE1_MEASURE"},
    {"a": ("products_sales_usd", "annual_sales_usd"), "b": ("products_sales_chf", "annual_sales_chf"), "label": "TYPE1_MEASURE"},
    {"a": ("products_sales_eur", "annual_sales_eur"), "b": ("products_sales_gbp", "annual_sales_gbp"), "label": "TYPE1_MEASURE"},
    {"a": ("products_sales_eur", "annual_sales_eur"), "b": ("products_sales_jpy", "annual_sales_jpy"), "label": "TYPE1_MEASURE"},
    {"a": ("products_sales_eur", "annual_sales_eur"), "b": ("products_sales_chf", "annual_sales_chf"), "label": "TYPE1_MEASURE"},
    {"a": ("products_sales_gbp", "annual_sales_gbp"), "b": ("products_sales_jpy", "annual_sales_jpy"), "label": "TYPE1_MEASURE"},
    {"a": ("products_sales_gbp", "annual_sales_gbp"), "b": ("products_sales_chf", "annual_sales_chf"), "label": "TYPE1_MEASURE"},
    {"a": ("products_sales_jpy", "annual_sales_jpy"), "b": ("products_sales_chf", "annual_sales_chf"), "label": "TYPE1_MEASURE"},
    # salary bands: USD vs EUR
    {"a": ("salary_bands_usd", "min_salary_usd"), "b": ("salary_bands_eur", "min_salary_eur"), "label": "TYPE1_MEASURE"},
    {"a": ("salary_bands_usd", "max_salary_usd"), "b": ("salary_bands_eur", "max_salary_eur"), "label": "TYPE1_MEASURE"},
    {"a": ("salary_bands_usd", "midpoint_usd"), "b": ("salary_bands_eur", "midpoint_eur"), "label": "TYPE1_MEASURE"},
    # ranking: raw pts vs normalized vs ELO
    {"a": ("sports_rankings_pts", "ranking_points"), "b": ("sports_rankings_norm", "points_norm"), "label": "TYPE1_MEASURE"},
    {"a": ("sports_rankings_pts", "ranking_points"), "b": ("sports_rankings_elo", "elo_rating"), "label": "TYPE1_MEASURE"},
    {"a": ("sports_rankings_norm", "points_norm"), "b": ("sports_rankings_elo", "elo_rating"), "label": "TYPE1_MEASURE"},
    # capacity: seats vs thousands
    {"a": ("venues_capacity_exact", "capacity_seats"), "b": ("venues_capacity_thousands", "capacity_thousands"), "label": "TYPE1_MEASURE"},
    {"a": ("venues_capacity_exact", "standing_capacity"), "b": ("venues_capacity_thousands", "standing_thousands"), "label": "TYPE1_MEASURE"},
    # price tiers: USD vs EUR label
    {"a": ("product_price_tiers_usd", "price_tier"), "b": ("product_price_tiers_eur", "price_tier_eur"), "label": "TYPE1_MEASURE"},
    # nationality encoding
    {"a": ("athletes_nat_full", "nationality"), "b": ("athletes_nat_iso2", "nationality_iso2"), "label": "TYPE1_MEASURE"},
    {"a": ("athletes_nat_full", "nationality"), "b": ("athletes_nat_iso3", "nationality_iso3"), "label": "TYPE1_MEASURE"},
    {"a": ("athletes_nat_iso2", "nationality_iso2"), "b": ("athletes_nat_iso3", "nationality_iso3"), "label": "TYPE1_MEASURE"},
    {"a": ("employees_nationality_full", "nationality"), "b": ("employees_nationality_iso2", "nationality_iso2"), "label": "TYPE1_MEASURE"},
    # gender encoding
    {"a": ("survey_gender_full", "gender"), "b": ("survey_gender_abbr", "gender_abbr"), "label": "TYPE1_MEASURE"},
    {"a": ("survey_gender_full", "gender"), "b": ("survey_gender_binary", "gender_code"), "label": "TYPE1_MEASURE"},
    {"a": ("survey_gender_abbr", "gender_abbr"), "b": ("survey_gender_binary", "gender_code"), "label": "TYPE1_MEASURE"},
    {"a": ("employees_gender_full", "gender"), "b": ("employees_gender_code", "gender_code"), "label": "TYPE1_MEASURE"},
    # name format variants
    {"a": ("persons_fullname", "full_name"), "b": ("persons_lastfirst", "last_first"), "label": "TYPE1_MEASURE"},
    {"a": ("persons_fullname", "full_name"), "b": ("persons_initials", "initials_name"), "label": "TYPE1_MEASURE"},
    {"a": ("persons_lastfirst", "last_first"), "b": ("persons_initials", "initials_name"), "label": "TYPE1_MEASURE"},
    {"a": ("employees_fullname", "full_name"), "b": ("employees_lastfirst", "last_first"), "label": "TYPE1_MEASURE"},
    {"a": ("authors_fullname", "full_name"), "b": ("authors_lastfirst", "last_first"), "label": "TYPE1_MEASURE"},
    # director format variants
    {"a": ("movies_directors", "director"), "b": ("movies_director_abbr", "director_abbr"), "label": "TYPE1_MEASURE"},
    {"a": ("movies_directors", "director"), "b": ("movies_director_lastfirst", "director_lastfirst"), "label": "TYPE1_MEASURE"},
    {"a": ("movies_director_abbr", "director_abbr"), "b": ("movies_director_lastfirst", "director_lastfirst"), "label": "TYPE1_MEASURE"},
    {"a": ("series_directors", "director"), "b": ("series_director_abbr", "director_abbr"), "label": "TYPE1_MEASURE"},
    # cast format variants
    {"a": ("movies_cast", "actor"), "b": ("movies_cast_abbr", "actor_abbr"), "label": "TYPE1_MEASURE"},
    # brand encoding
    {"a": ("products_brand_abbr", "brand_abbr"), "b": ("products_brand_full", "brand_full"), "label": "TYPE1_MEASURE"},
    {"a": ("retailers_brand_abbr", "brand_abbr"), "b": ("retailers_brand_full", "brand_full"), "label": "TYPE1_MEASURE"},

    # --- TYPE2_GRANULARITY: same concept, different aggregation level ---
    # sales temporal granularity
    {"a": ("products_sales_usd", "annual_sales_usd"), "b": ("products_monthly_usd", "monthly_sales_usd"), "label": "TYPE2_GRANULARITY"},
    {"a": ("products_sales_usd", "annual_sales_usd"), "b": ("products_quarterly_usd", "quarterly_sales_usd"), "label": "TYPE2_GRANULARITY"},
    {"a": ("products_sales_usd", "annual_sales_usd"), "b": ("products_weekly_usd", "weekly_sales_usd"), "label": "TYPE2_GRANULARITY"},
    {"a": ("products_monthly_usd", "monthly_sales_usd"), "b": ("products_quarterly_usd", "quarterly_sales_usd"), "label": "TYPE2_GRANULARITY"},
    {"a": ("products_monthly_usd", "monthly_sales_usd"), "b": ("products_weekly_usd", "weekly_sales_usd"), "label": "TYPE2_GRANULARITY"},
    {"a": ("products_quarterly_usd", "quarterly_sales_usd"), "b": ("products_weekly_usd", "weekly_sales_usd"), "label": "TYPE2_GRANULARITY"},
    # ranking temporal granularity
    {"a": ("sports_rankings_pts", "ranking_points"), "b": ("sports_rankings_monthly", "monthly_avg_points"), "label": "TYPE2_GRANULARITY"},
    {"a": ("sports_rankings_pts", "ranking_points"), "b": ("sports_rankings_season", "season_avg_points"), "label": "TYPE2_GRANULARITY"},
    {"a": ("sports_rankings_pts", "ranking_points"), "b": ("sports_rankings_career", "career_total_points"), "label": "TYPE2_GRANULARITY"},
    {"a": ("sports_rankings_monthly", "monthly_avg_points"), "b": ("sports_rankings_season", "season_avg_points"), "label": "TYPE2_GRANULARITY"},
    {"a": ("sports_rankings_monthly", "monthly_avg_points"), "b": ("sports_rankings_career", "career_total_points"), "label": "TYPE2_GRANULARITY"},
    {"a": ("sports_rankings_season", "season_avg_points"), "b": ("sports_rankings_career", "career_total_points"), "label": "TYPE2_GRANULARITY"},
    # venue attendance granularity
    {"a": ("venues_attendance_match", "attendance"), "b": ("venues_attendance_season", "avg_season_attendance"), "label": "TYPE2_GRANULARITY"},
    {"a": ("venues_attendance_match", "attendance"), "b": ("venues_attendance_annual", "total_annual_attendance"), "label": "TYPE2_GRANULARITY"},
    {"a": ("venues_attendance_season", "avg_season_attendance"), "b": ("venues_attendance_annual", "total_annual_attendance"), "label": "TYPE2_GRANULARITY"},
    # age distribution granularity
    {"a": ("age_dist_5yr", "population_count"), "b": ("age_dist_10yr", "population_count"), "label": "TYPE2_GRANULARITY"},
    {"a": ("age_dist_5yr", "age_range"), "b": ("age_dist_10yr", "age_range"), "label": "TYPE2_GRANULARITY"},
    # brand granularity: product-level vs summary
    {"a": ("products_brand_abbr", "annual_sales_usd"), "b": ("brand_category_summary", "total_sales_usd"), "label": "TYPE2_GRANULARITY"},
    {"a": ("products_brand_abbr", "annual_sales_usd"), "b": ("brand_category_summary", "avg_sales_usd"), "label": "TYPE2_GRANULARITY"},
    # cast granularity: per-movie vs summary
    {"a": ("movies_cast", "actor"), "b": ("cast_summary", "actor"), "label": "TYPE2_GRANULARITY"},
    # rank encoding: ordinal position vs percentile
    {"a": ("sports_rankings_pts", "world_rank"), "b": ("sports_rankings_norm", "rank_percentile"), "label": "TYPE2_GRANULARITY"},
    {"a": ("sports_rankings_pts", "world_rank"), "b": ("sports_rankings_elo", "rank_elo"), "label": "TYPE2_GRANULARITY"},
    # brand: product-level sales vs brand aggregate
    {"a": ("products_brand_abbr", "annual_sales_usd"), "b": ("brand_category_summary", "total_sales_usd"), "label": "TYPE2_GRANULARITY"},
    {"a": ("products_brand_abbr", "annual_sales_usd"), "b": ("brand_category_summary", "avg_sales_usd"), "label": "TYPE2_GRANULARITY"},
    # age: pct breakdown at different bin granularity
    {"a": ("age_dist_5yr", "pct_total"), "b": ("age_dist_10yr", "pct_total"), "label": "TYPE2_GRANULARITY"},
    # venue: match count at season vs annual grain
    {"a": ("venues_attendance_season", "match_count"), "b": ("venues_attendance_annual", "match_count"), "label": "TYPE2_GRANULARITY"},
    # movie year: per-movie vs per-movie-actor grain
    {"a": ("movies_directors", "year"), "b": ("movies_cast", "year"), "label": "TYPE2_GRANULARITY"},

    # --- NO_CONFLICT_DUPLICATE: same data, no semantic conflict ---
    # shared ID / name columns across format-variant tables
    {"a": ("products_sales_usd", "product_id"), "b": ("products_sales_eur", "product_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_sales_usd", "product_name"), "b": ("products_sales_gbp", "product_name"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_sales_usd", "brand"), "b": ("products_sales_jpy", "brand"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_sales_usd", "category"), "b": ("products_sales_chf", "category"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_monthly_usd", "product_id"), "b": ("products_quarterly_usd", "product_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("athletes_nat_full", "athlete_id"), "b": ("athletes_nat_iso2", "athlete_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("athletes_nat_full", "athlete_name"), "b": ("athletes_nat_iso3", "athlete_name"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("athletes_nat_full", "sport"), "b": ("athletes_nat_iso2", "sport"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("survey_gender_full", "respondent_id"), "b": ("survey_gender_abbr", "respondent_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("survey_gender_full", "age_group"), "b": ("survey_gender_binary", "age_group"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("survey_gender_full", "country"), "b": ("survey_gender_abbr", "country"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("persons_fullname", "person_id"), "b": ("persons_lastfirst", "person_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("persons_fullname", "department"), "b": ("persons_initials", "department"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("persons_fullname", "company"), "b": ("persons_lastfirst", "company"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("employees_fullname", "employee_id"), "b": ("employees_lastfirst", "employee_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("movies_directors", "movie_id"), "b": ("movies_director_abbr", "movie_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("movies_directors", "name"), "b": ("movies_director_lastfirst", "name"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("movies_directors", "year"), "b": ("movies_director_abbr", "year"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("movies_directors", "genre"), "b": ("movies_director_lastfirst", "genre"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("movies_cast", "movie_id"), "b": ("movies_cast_abbr", "movie_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("movies_cast", "name"), "b": ("movies_cast_abbr", "name"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("movies_cast", "year"), "b": ("movies_cast_abbr", "year"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("venues_capacity_exact", "venue_id"), "b": ("venues_capacity_thousands", "venue_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("venues_capacity_exact", "city"), "b": ("venues_capacity_thousands", "city"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("venues_capacity_exact", "country"), "b": ("venues_capacity_thousands", "country"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("venues_attendance_season", "venue_id"), "b": ("venues_attendance_annual", "venue_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("salary_bands_usd", "grade"), "b": ("salary_bands_eur", "grade"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("salary_bands_usd", "level"), "b": ("salary_bands_eur", "level"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("salary_bands_usd", "department"), "b": ("salary_bands_eur", "department"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("product_price_tiers_usd", "product_id"), "b": ("product_price_tiers_eur", "product_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("product_price_tiers_usd", "category"), "b": ("product_price_tiers_eur", "category"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_brand_abbr", "product_id"), "b": ("products_brand_full", "product_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_brand_abbr", "category"), "b": ("products_brand_full", "category"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("retailers_brand_abbr", "store_id"), "b": ("retailers_brand_full", "store_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("sports_rankings_pts", "player_id"), "b": ("sports_rankings_norm", "player_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("sports_rankings_pts", "player_name"), "b": ("sports_rankings_elo", "player_name"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("sports_rankings_pts", "sport"), "b": ("sports_rankings_monthly", "sport"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("authors_fullname", "author_id"), "b": ("authors_lastfirst", "author_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("authors_fullname", "genre"), "b": ("authors_lastfirst", "genre"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("employees_nationality_full", "employee_id"), "b": ("employees_nationality_iso2", "employee_id"), "label": "NO_CONFLICT_DUPLICATE"},

    # --- NO_CONFLICT_DIFF_ENTITY: different real-world concepts ---
    {"a": ("products_sales_usd", "annual_sales_usd"), "b": ("venues_attendance_season", "avg_season_attendance"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("products_sales_usd", "annual_sales_usd"), "b": ("sports_rankings_pts", "ranking_points"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("products_sales_eur", "annual_sales_eur"), "b": ("salary_bands_eur", "min_salary_eur"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("venues_capacity_exact", "capacity_seats"), "b": ("sports_rankings_pts", "ranking_points"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("venues_capacity_exact", "capacity_seats"), "b": ("products_sales_usd", "annual_sales_usd"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("movies_directors", "director"), "b": ("athletes_nat_full", "athlete_name"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("movies_directors", "director"), "b": ("authors_fullname", "full_name"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("series_directors", "director"), "b": ("movies_directors", "director"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("movies_cast", "actor"), "b": ("athletes_nat_full", "athlete_name"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("movies_cast", "actor"), "b": ("persons_fullname", "full_name"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("athletes_nat_full", "nationality"), "b": ("employees_nationality_full", "nationality"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("survey_gender_full", "gender"), "b": ("employees_gender_full", "gender"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("salary_bands_usd", "min_salary_usd"), "b": ("venues_capacity_exact", "capacity_seats"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("age_dist_5yr", "population_count"), "b": ("venues_attendance_annual", "total_annual_attendance"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("product_price_tiers_usd", "price_tier"), "b": ("age_dist_5yr", "age_range"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("products_brand_abbr", "brand_abbr"), "b": ("movies_directors", "director"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("retailers_brand_full", "brand_full"), "b": ("movies_director_abbr", "director_abbr"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("authors_fullname", "full_name"), "b": ("persons_fullname", "full_name"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("employees_fullname", "full_name"), "b": ("athletes_nat_full", "athlete_name"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("sports_rankings_career", "career_total_points"), "b": ("venues_attendance_annual", "total_annual_attendance"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("salary_bands_usd", "grade"), "b": ("product_price_tiers_usd", "price_tier"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("movies_directors", "genre"), "b": ("series_directors", "genre"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("venues_capacity_exact", "city"), "b": ("athletes_nat_full", "nationality"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("survey_gender_full", "age_group"), "b": ("age_dist_5yr", "age_range"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("persons_fullname", "company"), "b": ("products_brand_full", "brand_full"), "label": "NO_CONFLICT_DIFF_ENTITY"},

    # --- TYPE1_MEASURE: new tables ---
    # nationality: IOC vs ISO variants
    {"a": ("athletes_nat_full", "nationality"), "b": ("athletes_nat_ioc", "nationality_ioc"), "label": "TYPE1_MEASURE"},
    {"a": ("athletes_nat_iso2", "nationality_iso2"), "b": ("athletes_nat_ioc", "nationality_ioc"), "label": "TYPE1_MEASURE"},
    {"a": ("athletes_nat_iso3", "nationality_iso3"), "b": ("athletes_nat_ioc", "nationality_ioc"), "label": "TYPE1_MEASURE"},
    # sales: same temporal grain, different currency
    {"a": ("products_monthly_usd", "monthly_sales_usd"), "b": ("products_monthly_eur", "monthly_sales_eur"), "label": "TYPE1_MEASURE"},
    {"a": ("products_quarterly_usd", "quarterly_sales_usd"), "b": ("products_quarterly_eur", "quarterly_sales_eur"), "label": "TYPE1_MEASURE"},
    # salary: same grain, different currency
    {"a": ("employee_salary_usd", "annual_salary_usd"), "b": ("employee_salary_eur", "annual_salary_eur"), "label": "TYPE1_MEASURE"},
    {"a": ("employee_salary_usd", "monthly_salary_usd"), "b": ("employee_salary_eur", "monthly_salary_eur"), "label": "TYPE1_MEASURE"},
    # climate: same temporal grain, different unit
    {"a": ("city_climate_celsius", "avg_temp_celsius"), "b": ("city_climate_fahrenheit", "avg_temp_fahrenheit"), "label": "TYPE1_MEASURE"},
    {"a": ("city_climate_annual_celsius", "annual_avg_celsius"), "b": ("city_climate_annual_fahrenheit", "annual_avg_fahrenheit"), "label": "TYPE1_MEASURE"},
    # same-table column pairs: different representation of same concept
    {"a": ("sports_rankings_pts", "ranking_points"), "b": ("sports_rankings_pts", "world_rank"), "label": "TYPE1_MEASURE"},
    {"a": ("sports_rankings_norm", "points_norm"), "b": ("sports_rankings_norm", "rank_percentile"), "label": "TYPE1_MEASURE"},
    {"a": ("salary_bands_usd", "min_salary_usd"), "b": ("salary_bands_usd", "max_salary_usd"), "label": "TYPE1_MEASURE"},
    {"a": ("salary_bands_usd", "min_salary_usd"), "b": ("salary_bands_usd", "midpoint_usd"), "label": "TYPE1_MEASURE"},
    {"a": ("salary_bands_usd", "max_salary_usd"), "b": ("salary_bands_usd", "midpoint_usd"), "label": "TYPE1_MEASURE"},
    {"a": ("salary_bands_eur", "min_salary_eur"), "b": ("salary_bands_eur", "max_salary_eur"), "label": "TYPE1_MEASURE"},
    {"a": ("salary_bands_eur", "min_salary_eur"), "b": ("salary_bands_eur", "midpoint_eur"), "label": "TYPE1_MEASURE"},
    {"a": ("salary_bands_eur", "max_salary_eur"), "b": ("salary_bands_eur", "midpoint_eur"), "label": "TYPE1_MEASURE"},
    {"a": ("brand_category_summary", "brand"), "b": ("brand_category_summary", "brand_full"), "label": "TYPE1_MEASURE"},
    {"a": ("venues_capacity_exact", "capacity_seats"), "b": ("venues_capacity_exact", "standing_capacity"), "label": "TYPE1_MEASURE"},
    {"a": ("venues_capacity_thousands", "capacity_thousands"), "b": ("venues_capacity_thousands", "standing_thousands"), "label": "TYPE1_MEASURE"},

    # --- TYPE2_GRANULARITY: new tables ---
    # EUR sales temporal
    {"a": ("products_sales_eur", "annual_sales_eur"), "b": ("products_monthly_eur", "monthly_sales_eur"), "label": "TYPE2_GRANULARITY"},
    {"a": ("products_sales_eur", "annual_sales_eur"), "b": ("products_quarterly_eur", "quarterly_sales_eur"), "label": "TYPE2_GRANULARITY"},
    {"a": ("products_monthly_eur", "monthly_sales_eur"), "b": ("products_quarterly_eur", "quarterly_sales_eur"), "label": "TYPE2_GRANULARITY"},
    # employee salary: annual vs monthly grain
    {"a": ("employee_salary_usd", "annual_salary_usd"), "b": ("employee_salary_usd", "monthly_salary_usd"), "label": "TYPE2_GRANULARITY"},
    {"a": ("employee_salary_eur", "annual_salary_eur"), "b": ("employee_salary_eur", "monthly_salary_eur"), "label": "TYPE2_GRANULARITY"},
    # employee salary vs grade band: per-person vs per-grade
    {"a": ("employee_salary_usd", "annual_salary_usd"), "b": ("salary_bands_usd", "midpoint_usd"), "label": "TYPE2_GRANULARITY"},
    {"a": ("employee_salary_eur", "annual_salary_eur"), "b": ("salary_bands_eur", "midpoint_eur"), "label": "TYPE2_GRANULARITY"},
    # climate: monthly vs annual
    {"a": ("city_climate_celsius", "avg_temp_celsius"), "b": ("city_climate_annual_celsius", "annual_avg_celsius"), "label": "TYPE2_GRANULARITY"},
    {"a": ("city_climate_fahrenheit", "avg_temp_fahrenheit"), "b": ("city_climate_annual_fahrenheit", "annual_avg_fahrenheit"), "label": "TYPE2_GRANULARITY"},
    # brand: product-level total vs aggregate average
    {"a": ("brand_category_summary", "total_sales_usd"), "b": ("brand_category_summary", "avg_sales_usd"), "label": "TYPE2_GRANULARITY"},

    # --- NO_CONFLICT_DUPLICATE: new tables ---
    {"a": ("athletes_nat_full", "athlete_id"), "b": ("athletes_nat_ioc", "athlete_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("athletes_nat_full", "athlete_name"), "b": ("athletes_nat_ioc", "athlete_name"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("athletes_nat_full", "sport"), "b": ("athletes_nat_ioc", "sport"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("athletes_nat_iso2", "athlete_id"), "b": ("athletes_nat_ioc", "athlete_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_monthly_eur", "product_id"), "b": ("products_quarterly_eur", "product_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_monthly_eur", "product_name"), "b": ("products_quarterly_eur", "product_name"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_monthly_eur", "brand"), "b": ("products_quarterly_eur", "brand"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_monthly_eur", "product_id"), "b": ("products_monthly_usd", "product_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_quarterly_eur", "product_id"), "b": ("products_quarterly_usd", "product_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("products_monthly_eur", "category"), "b": ("products_sales_eur", "category"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("employee_salary_usd", "employee_id"), "b": ("employee_salary_eur", "employee_id"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("employee_salary_usd", "full_name"), "b": ("employee_salary_eur", "full_name"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("employee_salary_usd", "department"), "b": ("employee_salary_eur", "department"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("employee_salary_usd", "grade"), "b": ("employee_salary_eur", "grade"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("employee_salary_usd", "level"), "b": ("employee_salary_eur", "level"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("city_climate_celsius", "city"), "b": ("city_climate_fahrenheit", "city"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("city_climate_celsius", "country"), "b": ("city_climate_fahrenheit", "country"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("city_climate_celsius", "month"), "b": ("city_climate_fahrenheit", "month"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("city_climate_celsius", "city"), "b": ("city_climate_annual_celsius", "city"), "label": "NO_CONFLICT_DUPLICATE"},
    {"a": ("city_climate_annual_celsius", "city"), "b": ("city_climate_annual_fahrenheit", "city"), "label": "NO_CONFLICT_DUPLICATE"},

    # --- NO_CONFLICT_DIFF_ENTITY: new tables ---
    {"a": ("employee_salary_usd", "annual_salary_usd"), "b": ("products_sales_usd", "annual_sales_usd"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("employee_salary_usd", "annual_salary_usd"), "b": ("sports_rankings_pts", "ranking_points"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("employee_salary_usd", "annual_salary_usd"), "b": ("venues_capacity_exact", "capacity_seats"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("city_climate_celsius", "avg_temp_celsius"), "b": ("venues_attendance_season", "avg_season_attendance"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("city_climate_celsius", "avg_temp_celsius"), "b": ("products_sales_usd", "annual_sales_usd"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("city_climate_annual_celsius", "annual_avg_celsius"), "b": ("salary_bands_usd", "midpoint_usd"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("city_climate_fahrenheit", "avg_temp_fahrenheit"), "b": ("sports_rankings_pts", "ranking_points"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("employee_salary_usd", "full_name"), "b": ("series_directors", "director"), "label": "NO_CONFLICT_DIFF_ENTITY"},
    {"a": ("employee_salary_usd", "grade"), "b": ("product_price_tiers_usd", "price_tier"), "label": "NO_CONFLICT_DIFF_ENTITY"},
]


DDL_TEMPLATES: dict[str, str] = {
    "products_sales_usd": "CREATE TABLE products_sales_usd (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    annual_sales_usd BIGINT COMMENT 'Annual sales in US dollars',\n    units_sold BIGINT COMMENT 'Total units sold'\n)\nCOMMENT 'Annual product sales in USD'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='USD' grain='per-product-annual');",
    "products_sales_eur": "CREATE TABLE products_sales_eur (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    annual_sales_eur BIGINT COMMENT 'Annual sales in euros',\n    units_sold BIGINT COMMENT 'Total units sold'\n)\nCOMMENT 'Annual product sales in EUR'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='EUR' grain='per-product-annual');",
    "products_sales_gbp": "CREATE TABLE products_sales_gbp (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    annual_sales_gbp BIGINT COMMENT 'Annual sales in British pounds',\n    units_sold BIGINT COMMENT 'Total units sold'\n)\nCOMMENT 'Annual product sales in GBP'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='GBP' grain='per-product-annual');",
    "products_sales_jpy": "CREATE TABLE products_sales_jpy (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    annual_sales_jpy BIGINT COMMENT 'Annual sales in Japanese yen',\n    units_sold BIGINT COMMENT 'Total units sold'\n)\nCOMMENT 'Annual product sales in JPY'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='JPY' grain='per-product-annual');",
    "products_sales_chf": "CREATE TABLE products_sales_chf (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    annual_sales_chf BIGINT COMMENT 'Annual sales in Swiss francs',\n    units_sold BIGINT COMMENT 'Total units sold'\n)\nCOMMENT 'Annual product sales in CHF'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='CHF' grain='per-product-annual');",
    "products_monthly_usd": "CREATE TABLE products_monthly_usd (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    report_month STRING COMMENT 'Reporting month (YYYY-MM)',\n    monthly_sales_usd BIGINT COMMENT 'Monthly sales in USD'\n)\nCOMMENT 'Monthly product sales in USD'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='USD' grain='per-product-monthly');",
    "products_quarterly_usd": "CREATE TABLE products_quarterly_usd (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    report_quarter STRING COMMENT 'Reporting quarter (Q1-2023)',\n    quarterly_sales_usd BIGINT COMMENT 'Quarterly sales in USD'\n)\nCOMMENT 'Quarterly product sales in USD'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='USD' grain='per-product-quarterly');",
    "products_weekly_usd": "CREATE TABLE products_weekly_usd (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    report_week STRING COMMENT 'ISO week (2023-W01)',\n    weekly_sales_usd BIGINT COMMENT 'Weekly sales in USD'\n)\nCOMMENT 'Weekly product sales in USD'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='USD' grain='per-product-weekly');",
    "products_brand_abbr": "CREATE TABLE products_brand_abbr (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand_abbr STRING COMMENT 'Brand abbreviation or common short name',\n    category STRING COMMENT 'Product category',\n    annual_sales_usd BIGINT COMMENT 'Annual sales in USD'\n)\nCOMMENT 'Product catalog with abbreviated brand names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-product');",
    "products_brand_full": "CREATE TABLE products_brand_full (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand_full STRING COMMENT 'Full legal brand name',\n    category STRING COMMENT 'Product category',\n    annual_sales_usd BIGINT COMMENT 'Annual sales in USD'\n)\nCOMMENT 'Product catalog with full legal brand names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-legal' grain='per-product');",
    "brand_category_summary": "CREATE TABLE brand_category_summary (\n    brand STRING COMMENT 'Brand abbreviation',\n    brand_full STRING COMMENT 'Full legal brand name',\n    total_products INT COMMENT 'Number of products in dataset',\n    total_sales_usd BIGINT COMMENT 'Total sales across all products in USD',\n    avg_sales_usd BIGINT COMMENT 'Average sales per product in USD'\n)\nCOMMENT 'Per-brand aggregate sales summary'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' grain='per-brand');",
    "retailers_brand_abbr": "CREATE TABLE retailers_brand_abbr (\n    store_id STRING COMMENT 'Unique store identifier',\n    store_name STRING COMMENT 'Store name',\n    brand_abbr STRING COMMENT 'Retailer brand abbreviation',\n    country STRING COMMENT 'Country of operation',\n    annual_revenue_eur BIGINT COMMENT 'Annual store revenue in EUR'\n)\nCOMMENT 'Retail store catalog with abbreviated brand names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-store');",
    "retailers_brand_full": "CREATE TABLE retailers_brand_full (\n    store_id STRING COMMENT 'Unique store identifier',\n    store_name STRING COMMENT 'Store name',\n    brand_full STRING COMMENT 'Retailer full legal brand name',\n    country STRING COMMENT 'Country of operation',\n    annual_revenue_eur BIGINT COMMENT 'Annual store revenue in EUR'\n)\nCOMMENT 'Retail store catalog with full legal brand names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-legal' grain='per-store');",
    "sports_rankings_pts": "CREATE TABLE sports_rankings_pts (\n    player_id STRING COMMENT 'Unique player identifier',\n    player_name STRING COMMENT 'Player full name',\n    nationality STRING COMMENT 'Nationality full country name',\n    sport STRING COMMENT 'Sport discipline',\n    ranking_points INT COMMENT 'Raw ranking points',\n    world_rank INT COMMENT 'World ranking position'\n)\nCOMMENT 'Player rankings in raw points'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' scoring='raw-points' grain='per-player');",
    "sports_rankings_norm": "CREATE TABLE sports_rankings_norm (\n    player_id STRING COMMENT 'Unique player identifier',\n    player_name STRING COMMENT 'Player full name',\n    nationality STRING COMMENT 'Nationality',\n    sport STRING COMMENT 'Sport discipline',\n    points_norm DOUBLE COMMENT 'Normalised ranking score (0-100 scale)',\n    rank_percentile DOUBLE COMMENT 'Rank percentile (1.0 = top)'\n)\nCOMMENT 'Player rankings normalised to 0-100 scale'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' scoring='normalised' grain='per-player');",
    "sports_rankings_elo": "CREATE TABLE sports_rankings_elo (\n    player_id STRING COMMENT 'Unique player identifier',\n    player_name STRING COMMENT 'Player full name',\n    nationality STRING COMMENT 'Nationality',\n    sport STRING COMMENT 'Sport discipline',\n    elo_rating INT COMMENT 'ELO-style rating (baseline 1200)',\n    rank_elo INT COMMENT 'Rank by ELO rating'\n)\nCOMMENT 'Player rankings using ELO rating system'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' scoring='elo' grain='per-player');",
    "sports_rankings_monthly": "CREATE TABLE sports_rankings_monthly (\n    player_id STRING COMMENT 'Unique player identifier',\n    player_name STRING COMMENT 'Player full name',\n    sport STRING COMMENT 'Sport discipline',\n    report_month STRING COMMENT 'Reporting month (YYYY-MM)',\n    monthly_avg_points INT COMMENT 'Average monthly points'\n)\nCOMMENT 'Monthly average player ranking points'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' grain='per-player-monthly');",
    "sports_rankings_season": "CREATE TABLE sports_rankings_season (\n    player_id STRING COMMENT 'Unique player identifier',\n    player_name STRING COMMENT 'Player full name',\n    sport STRING COMMENT 'Sport discipline',\n    season STRING COMMENT 'Season identifier (2022-2023)',\n    season_avg_points INT COMMENT 'Season average points'\n)\nCOMMENT 'Season average player ranking points'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' grain='per-player-season');",
    "sports_rankings_career": "CREATE TABLE sports_rankings_career (\n    player_id STRING COMMENT 'Unique player identifier',\n    player_name STRING COMMENT 'Player full name',\n    sport STRING COMMENT 'Sport discipline',\n    career_span STRING COMMENT 'Career period (2015-2023)',\n    career_total_points INT COMMENT 'Career total points accumulated'\n)\nCOMMENT 'Career total player ranking points'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' grain='per-player-career');",
    "venues_capacity_exact": "CREATE TABLE venues_capacity_exact (\n    venue_id STRING COMMENT 'Unique venue identifier',\n    venue_name STRING COMMENT 'Venue name',\n    city STRING COMMENT 'City',\n    country STRING COMMENT 'Country',\n    capacity_seats INT COMMENT 'Seating capacity in individual seats',\n    standing_capacity INT COMMENT 'Standing area capacity'\n)\nCOMMENT 'Venue capacity in exact seat count'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' unit='seats' grain='per-venue');",
    "venues_capacity_thousands": "CREATE TABLE venues_capacity_thousands (\n    venue_id STRING COMMENT 'Unique venue identifier',\n    venue_name STRING COMMENT 'Venue name',\n    city STRING COMMENT 'City',\n    country STRING COMMENT 'Country',\n    capacity_thousands DOUBLE COMMENT 'Seating capacity in thousands of seats',\n    standing_thousands DOUBLE COMMENT 'Standing capacity in thousands'\n)\nCOMMENT 'Venue capacity in thousands of seats'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' unit='thousands' grain='per-venue');",
    "venues_attendance_match": "CREATE TABLE venues_attendance_match (\n    match_id STRING COMMENT 'Unique match identifier',\n    venue_id STRING COMMENT 'Venue identifier',\n    venue_name STRING COMMENT 'Venue name',\n    event_date STRING COMMENT 'Event date (YYYY-MM-DD)',\n    attendance INT COMMENT 'Per-match attendance count'\n)\nCOMMENT 'Per-match attendance records'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' grain='per-match');",
    "venues_attendance_season": "CREATE TABLE venues_attendance_season (\n    venue_id STRING COMMENT 'Venue identifier',\n    venue_name STRING COMMENT 'Venue name',\n    season STRING COMMENT 'Season identifier',\n    avg_season_attendance INT COMMENT 'Average attendance per match in the season',\n    match_count INT COMMENT 'Number of matches in the season'\n)\nCOMMENT 'Season average attendance per venue'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' grain='per-venue-season');",
    "venues_attendance_annual": "CREATE TABLE venues_attendance_annual (\n    venue_id STRING COMMENT 'Venue identifier',\n    venue_name STRING COMMENT 'Venue name',\n    year INT COMMENT 'Calendar year',\n    total_annual_attendance INT COMMENT 'Total attendance across all matches in the year',\n    match_count INT COMMENT 'Number of matches held'\n)\nCOMMENT 'Annual total attendance per venue'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' grain='per-venue-annual');",
    "athletes_nat_full": "CREATE TABLE athletes_nat_full (\n    athlete_id STRING COMMENT 'Unique athlete identifier',\n    athlete_name STRING COMMENT 'Athlete full name',\n    sport STRING COMMENT 'Sport discipline',\n    nationality STRING COMMENT 'Nationality full country name'\n)\nCOMMENT 'Athlete roster with full nationality names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-country-name' grain='per-athlete');",
    "athletes_nat_iso2": "CREATE TABLE athletes_nat_iso2 (\n    athlete_id STRING COMMENT 'Athlete identifier',\n    athlete_name STRING COMMENT 'Athlete name',\n    sport STRING COMMENT 'Sport',\n    nationality_iso2 STRING COMMENT 'ISO 3166-1 alpha-2 nationality code (e.g. ES)'\n)\nCOMMENT 'Athlete roster with ISO 3166-1 alpha-2 nationality codes'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='ISO-3166-1-alpha-2' grain='per-athlete');",
    "athletes_nat_iso3": "CREATE TABLE athletes_nat_iso3 (\n    athlete_id STRING COMMENT 'Athlete identifier',\n    athlete_name STRING COMMENT 'Athlete name',\n    sport STRING COMMENT 'Sport',\n    nationality_iso3 STRING COMMENT 'ISO 3166-1 alpha-3 nationality code (e.g. ESP)'\n)\nCOMMENT 'Athlete roster with ISO 3166-1 alpha-3 nationality codes'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='ISO-3166-1-alpha-3' grain='per-athlete');",
    "employees_nationality_full": "CREATE TABLE employees_nationality_full (\n    employee_id STRING COMMENT 'Unique employee identifier',\n    full_name STRING COMMENT 'Employee full name',\n    department STRING COMMENT 'Department',\n    nationality STRING COMMENT 'Nationality full country name'\n)\nCOMMENT 'Employee roster with full nationality names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-country-name' grain='per-employee');",
    "employees_nationality_iso2": "CREATE TABLE employees_nationality_iso2 (\n    employee_id STRING COMMENT 'Unique employee identifier',\n    full_name STRING COMMENT 'Employee full name',\n    department STRING COMMENT 'Department',\n    nationality_iso2 STRING COMMENT 'ISO 3166-1 alpha-2 nationality code'\n)\nCOMMENT 'Employee roster with ISO 3166-1 alpha-2 nationality codes'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='ISO-3166-1-alpha-2' grain='per-employee');",
    "survey_gender_full": "CREATE TABLE survey_gender_full (\n    respondent_id STRING COMMENT 'Unique respondent identifier',\n    age_group STRING COMMENT 'Age group range (e.g. 25-34)',\n    country STRING COMMENT 'Country of respondent',\n    gender STRING COMMENT 'Gender full text (Male / Female / Non-binary)'\n)\nCOMMENT 'Survey responses with full-text gender values'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-text' grain='per-respondent');",
    "survey_gender_abbr": "CREATE TABLE survey_gender_abbr (\n    respondent_id STRING COMMENT 'Unique respondent identifier',\n    age_group STRING COMMENT 'Age group range',\n    country STRING COMMENT 'Country',\n    gender_abbr STRING COMMENT 'Gender abbreviation (M / F / NB)'\n)\nCOMMENT 'Survey responses with abbreviated gender codes'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-respondent');",
    "survey_gender_binary": "CREATE TABLE survey_gender_binary (\n    respondent_id STRING COMMENT 'Unique respondent identifier',\n    age_group STRING COMMENT 'Age group range',\n    country STRING COMMENT 'Country',\n    gender_code INT COMMENT 'Gender numeric code (0=Female 1=Male 2=Non-binary)'\n)\nCOMMENT 'Survey responses with numeric gender codes'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='numeric' grain='per-respondent');",
    "employees_gender_full": "CREATE TABLE employees_gender_full (\n    employee_id STRING COMMENT 'Unique employee identifier',\n    department STRING COMMENT 'Department',\n    gender STRING COMMENT 'Gender full text'\n)\nCOMMENT 'Employee gender with full-text values'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-text' grain='per-employee');",
    "employees_gender_code": "CREATE TABLE employees_gender_code (\n    employee_id STRING COMMENT 'Unique employee identifier',\n    department STRING COMMENT 'Department',\n    gender_code INT COMMENT 'Gender numeric code (0=Female 1=Male 2=Non-binary)'\n)\nCOMMENT 'Employee gender with numeric codes'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='numeric' grain='per-employee');",
    "persons_fullname": "CREATE TABLE persons_fullname (\n    person_id STRING COMMENT 'Unique person identifier',\n    full_name STRING COMMENT 'Full name (Firstname Lastname)',\n    department STRING COMMENT 'Department',\n    company STRING COMMENT 'Company'\n)\nCOMMENT 'Person directory with full names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-person');",
    "persons_lastfirst": "CREATE TABLE persons_lastfirst (\n    person_id STRING COMMENT 'Unique person identifier',\n    last_first STRING COMMENT 'Name in Lastname, Firstname format',\n    department STRING COMMENT 'Department',\n    company STRING COMMENT 'Company'\n)\nCOMMENT 'Person directory with last-first name format'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='last-first' grain='per-person');",
    "persons_initials": "CREATE TABLE persons_initials (\n    person_id STRING COMMENT 'Unique person identifier',\n    initials_name STRING COMMENT 'Name with abbreviated first name (F. Lastname)',\n    department STRING COMMENT 'Department',\n    company STRING COMMENT 'Company'\n)\nCOMMENT 'Person directory with abbreviated first name'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-person');",
    "employees_fullname": "CREATE TABLE employees_fullname (\n    employee_id STRING COMMENT 'Unique employee identifier',\n    full_name STRING COMMENT 'Full name (Firstname Lastname)',\n    department STRING COMMENT 'Department',\n    company STRING COMMENT 'Company'\n)\nCOMMENT 'Employee directory with full names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-employee');",
    "employees_lastfirst": "CREATE TABLE employees_lastfirst (\n    employee_id STRING COMMENT 'Unique employee identifier',\n    last_first STRING COMMENT 'Name in Lastname, Firstname format',\n    department STRING COMMENT 'Department',\n    company STRING COMMENT 'Company'\n)\nCOMMENT 'Employee directory with last-first name format'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='last-first' grain='per-employee');",
    "authors_fullname": "CREATE TABLE authors_fullname (\n    author_id STRING COMMENT 'Unique author identifier',\n    full_name STRING COMMENT 'Author full name',\n    genre STRING COMMENT 'Primary writing genre',\n    country STRING COMMENT 'Country of origin'\n)\nCOMMENT 'Author directory with full names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-author');",
    "authors_lastfirst": "CREATE TABLE authors_lastfirst (\n    author_id STRING COMMENT 'Unique author identifier',\n    last_first STRING COMMENT 'Author name in Lastname, Firstname format',\n    genre STRING COMMENT 'Primary writing genre',\n    country STRING COMMENT 'Country of origin'\n)\nCOMMENT 'Author directory with last-first name format'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='last-first' grain='per-author');",
    "movies_directors": "CREATE TABLE movies_directors (\n    movie_id STRING COMMENT 'Unique movie identifier',\n    name STRING COMMENT 'Movie title',\n    director STRING COMMENT 'Director full name',\n    year INT COMMENT 'Release year',\n    language STRING COMMENT 'Original language code',\n    genre STRING COMMENT 'Primary genre'\n)\nCOMMENT 'Movie catalog with director full names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-movie');",
    "movies_director_abbr": "CREATE TABLE movies_director_abbr (\n    movie_id STRING COMMENT 'Unique movie identifier',\n    name STRING COMMENT 'Movie title',\n    director_abbr STRING COMMENT 'Director name abbreviated (F. Lastname)',\n    year INT COMMENT 'Release year',\n    language STRING COMMENT 'Original language code',\n    genre STRING COMMENT 'Primary genre'\n)\nCOMMENT 'Movie catalog with abbreviated director names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-movie');",
    "movies_director_lastfirst": "CREATE TABLE movies_director_lastfirst (\n    movie_id STRING COMMENT 'Unique movie identifier',\n    name STRING COMMENT 'Movie title',\n    director_lastfirst STRING COMMENT 'Director name in Lastname, Firstname format',\n    year INT COMMENT 'Release year',\n    language STRING COMMENT 'Original language code',\n    genre STRING COMMENT 'Primary genre'\n)\nCOMMENT 'Movie catalog with last-first director names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='last-first' grain='per-movie');",
    "movies_cast": "CREATE TABLE movies_cast (\n    movie_id STRING COMMENT 'Movie identifier',\n    name STRING COMMENT 'Movie title',\n    year INT COMMENT 'Release year',\n    actor STRING COMMENT 'Actor full name'\n)\nCOMMENT 'Movie cast with actor full names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-movie-actor');",
    "movies_cast_abbr": "CREATE TABLE movies_cast_abbr (\n    movie_id STRING COMMENT 'Movie identifier',\n    name STRING COMMENT 'Movie title',\n    year INT COMMENT 'Release year',\n    actor_abbr STRING COMMENT 'Actor name abbreviated (F. Lastname)'\n)\nCOMMENT 'Movie cast with abbreviated actor names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-movie-actor');",
    "cast_summary": "CREATE TABLE cast_summary (\n    actor STRING COMMENT 'Actor full name',\n    total_movies INT COMMENT 'Number of movies the actor appeared in'\n)\nCOMMENT 'Per-actor movie count summary'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' grain='per-actor');",
    "series_directors": "CREATE TABLE series_directors (\n    series_id STRING COMMENT 'Unique series identifier',\n    name STRING COMMENT 'Series title',\n    director STRING COMMENT 'Director full name',\n    year INT COMMENT 'Release year',\n    language STRING COMMENT 'Original language code',\n    genre STRING COMMENT 'Primary genre'\n)\nCOMMENT 'TV series catalog with director full names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-series');",
    "series_director_abbr": "CREATE TABLE series_director_abbr (\n    series_id STRING COMMENT 'Unique series identifier',\n    name STRING COMMENT 'Series title',\n    director_abbr STRING COMMENT 'Director name abbreviated (F. Lastname)',\n    year INT COMMENT 'Release year',\n    language STRING COMMENT 'Original language code',\n    genre STRING COMMENT 'Primary genre'\n)\nCOMMENT 'TV series catalog with abbreviated director names'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-series');",
    "salary_bands_usd": "CREATE TABLE salary_bands_usd (\n    department STRING COMMENT 'Department name',\n    grade STRING COMMENT 'Compensation grade (e.g. IC3)',\n    level STRING COMMENT 'Level label (e.g. Senior)',\n    min_salary_usd INT COMMENT 'Minimum salary for grade in USD',\n    max_salary_usd INT COMMENT 'Maximum salary for grade in USD',\n    midpoint_usd INT COMMENT 'Midpoint salary for grade in USD'\n)\nCOMMENT 'Salary band ranges per grade in USD'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='USD' grain='per-department-grade');",
    "salary_bands_eur": "CREATE TABLE salary_bands_eur (\n    department STRING COMMENT 'Department name',\n    grade STRING COMMENT 'Compensation grade (e.g. IC3)',\n    level STRING COMMENT 'Level label (e.g. Senior)',\n    min_salary_eur INT COMMENT 'Minimum salary for grade in EUR',\n    max_salary_eur INT COMMENT 'Maximum salary for grade in EUR',\n    midpoint_eur INT COMMENT 'Midpoint salary for grade in EUR'\n)\nCOMMENT 'Salary band ranges per grade in EUR'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='EUR' grain='per-department-grade');",
    "age_dist_5yr": "CREATE TABLE age_dist_5yr (\n    age_range STRING COMMENT 'Five-year age group (e.g. 25-29)',\n    population_count BIGINT COMMENT 'Population count in age group',\n    pct_total DOUBLE COMMENT 'Percentage of total population'\n)\nCOMMENT 'Population age distribution in 5-year bins'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' granularity='5yr' grain='per-age-bin');",
    "age_dist_10yr": "CREATE TABLE age_dist_10yr (\n    age_range STRING COMMENT 'Ten-year age group (e.g. 20-29)',\n    population_count BIGINT COMMENT 'Population count in age group',\n    pct_total DOUBLE COMMENT 'Percentage of total population'\n)\nCOMMENT 'Population age distribution in 10-year bins'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' granularity='10yr' grain='per-age-bin');",
    "product_price_tiers_usd": "CREATE TABLE product_price_tiers_usd (\n    product_id STRING COMMENT 'Unique product identifier',\n    category STRING COMMENT 'Product category',\n    price_tier STRING COMMENT 'Price tier label with USD range (e.g. Mid-range (100-250))',\n    brand STRING COMMENT 'Brand name'\n)\nCOMMENT 'Product price tier classification in USD ranges'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='USD' grain='per-product');",
    "product_price_tiers_eur": "CREATE TABLE product_price_tiers_eur (\n    product_id STRING COMMENT 'Unique product identifier',\n    category STRING COMMENT 'Product category',\n    price_tier_eur STRING COMMENT 'Price tier label with EUR range (e.g. Mid-range (91-228))',\n    brand STRING COMMENT 'Brand name'\n)\nCOMMENT 'Product price tier classification in EUR ranges'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='EUR' grain='per-product');",
    "athletes_nat_ioc": "CREATE TABLE athletes_nat_ioc (\n    athlete_id STRING COMMENT 'Unique athlete identifier',\n    athlete_name STRING COMMENT 'Athlete full name',\n    sport STRING COMMENT 'Sport discipline',\n    nationality_ioc STRING COMMENT 'IOC 3-letter nationality code (differs from ISO3 for 8 countries: GER/NED/SUI/GRE/DEN/CRO/POR/RSA)'\n)\nCOMMENT 'Athlete roster with IOC nationality codes used in Olympic competition'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' encoding='IOC-3letter' grain='per-athlete');",
    "products_monthly_eur": "CREATE TABLE products_monthly_eur (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    report_month STRING COMMENT 'Reporting month (YYYY-MM)',\n    monthly_sales_eur BIGINT COMMENT 'Monthly sales in euros'\n)\nCOMMENT 'Monthly product sales in EUR'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='EUR' grain='per-product-monthly');",
    "products_quarterly_eur": "CREATE TABLE products_quarterly_eur (\n    product_id STRING COMMENT 'Unique product identifier',\n    product_name STRING COMMENT 'Product name',\n    brand STRING COMMENT 'Brand name',\n    category STRING COMMENT 'Product category',\n    report_quarter STRING COMMENT 'Reporting quarter (Q1-2023)',\n    quarterly_sales_eur BIGINT COMMENT 'Quarterly sales in euros'\n)\nCOMMENT 'Quarterly product sales in EUR'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='EUR' grain='per-product-quarterly');",
    "employee_salary_usd": "CREATE TABLE employee_salary_usd (\n    employee_id STRING COMMENT 'Unique employee identifier',\n    full_name STRING COMMENT 'Employee full name',\n    department STRING COMMENT 'Department',\n    grade STRING COMMENT 'Compensation grade (e.g. IC3)',\n    level STRING COMMENT 'Level label (e.g. Senior)',\n    annual_salary_usd INT COMMENT 'Annual base salary in USD',\n    monthly_salary_usd INT COMMENT 'Monthly base salary in USD (annual / 12)'\n)\nCOMMENT 'Individual employee salaries in USD with annual and monthly grain'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='USD' grain='per-employee');",
    "employee_salary_eur": "CREATE TABLE employee_salary_eur (\n    employee_id STRING COMMENT 'Unique employee identifier',\n    full_name STRING COMMENT 'Employee full name',\n    department STRING COMMENT 'Department',\n    grade STRING COMMENT 'Compensation grade (e.g. IC3)',\n    level STRING COMMENT 'Level label (e.g. Senior)',\n    annual_salary_eur INT COMMENT 'Annual base salary in EUR',\n    monthly_salary_eur INT COMMENT 'Monthly base salary in EUR (annual / 12)'\n)\nCOMMENT 'Individual employee salaries in EUR with annual and monthly grain'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' currency='EUR' grain='per-employee');",
    "city_climate_celsius": "CREATE TABLE city_climate_celsius (\n    city STRING COMMENT 'City name',\n    country STRING COMMENT 'Country',\n    month INT COMMENT 'Month number (1-12)',\n    avg_temp_celsius DOUBLE COMMENT 'Average monthly temperature in degrees Celsius'\n)\nCOMMENT 'Monthly average city temperatures in Celsius'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' unit='celsius' grain='per-city-month');",
    "city_climate_fahrenheit": "CREATE TABLE city_climate_fahrenheit (\n    city STRING COMMENT 'City name',\n    country STRING COMMENT 'Country',\n    month INT COMMENT 'Month number (1-12)',\n    avg_temp_fahrenheit DOUBLE COMMENT 'Average monthly temperature in degrees Fahrenheit'\n)\nCOMMENT 'Monthly average city temperatures in Fahrenheit'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' unit='fahrenheit' grain='per-city-month');",
    "city_climate_annual_celsius": "CREATE TABLE city_climate_annual_celsius (\n    city STRING COMMENT 'City name',\n    country STRING COMMENT 'Country',\n    annual_avg_celsius DOUBLE COMMENT 'Annual average temperature in degrees Celsius'\n)\nCOMMENT 'Annual average city temperatures in Celsius'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' unit='celsius' grain='per-city-annual');",
    "city_climate_annual_fahrenheit": "CREATE TABLE city_climate_annual_fahrenheit (\n    city STRING COMMENT 'City name',\n    country STRING COMMENT 'Country',\n    annual_avg_fahrenheit DOUBLE COMMENT 'Annual average temperature in degrees Fahrenheit'\n)\nCOMMENT 'Annual average city temperatures in Fahrenheit'\nSTORED AS PARQUET\nTBLPROPERTIES (source='synthetic' unit='fahrenheit' grain='per-city-annual');",
}

LINEAGE_TEMPLATES: list[dict] = [
    {"job": "ingest_products_sales", "inputs": ["source/synthetic-products"], "outputs": [
        "products_sales_usd", "products_sales_eur", "products_sales_gbp",
        "products_sales_jpy", "products_sales_chf",
    ], "sql_tmpl": "SELECT product_id, product_name, brand, category, annual_sales_{cur} FROM synthetic_products"},
    {"job": "aggregate_products_temporal", "inputs": ["products_sales_usd"], "outputs": [
        "products_monthly_usd", "products_quarterly_usd", "products_weekly_usd",
    ], "sql_tmpl": "SELECT product_id, product_name, brand, category, {period} AS report_{grain}, ROUND(annual_sales_usd / {divisor}, 0) AS {col} FROM products_sales_usd"},
    {"job": "ingest_brands", "inputs": ["source/synthetic-products"], "outputs": [
        "products_brand_abbr", "products_brand_full",
    ], "sql_tmpl": "SELECT product_id, product_name, {brand_col} AS {col_alias}, category, annual_sales_usd FROM synthetic_products"},
    {"job": "aggregate_brand_summary", "inputs": ["products_brand_abbr"], "outputs": ["brand_category_summary"],
     "sql_tmpl": "SELECT brand_abbr AS brand, brand_full, COUNT(*) AS total_products, SUM(annual_sales_usd) AS total_sales_usd, AVG(annual_sales_usd) AS avg_sales_usd FROM products_brand_abbr GROUP BY brand_abbr, brand_full"},
    {"job": "ingest_retailers", "inputs": ["source/synthetic-retailers"], "outputs": [
        "retailers_brand_abbr", "retailers_brand_full",
    ], "sql_tmpl": "SELECT store_id, store_name, {brand_col} AS {col_alias}, country, annual_revenue_eur FROM synthetic_retailers"},
    {"job": "ingest_sports_rankings", "inputs": ["source/synthetic-rankings"], "outputs": [
        "sports_rankings_pts", "sports_rankings_norm", "sports_rankings_elo",
    ], "sql_tmpl": "SELECT player_id, player_name, nationality, sport, {scoring_col} FROM synthetic_rankings"},
    {"job": "aggregate_rankings_temporal", "inputs": ["sports_rankings_pts"], "outputs": [
        "sports_rankings_monthly", "sports_rankings_season", "sports_rankings_career",
    ], "sql_tmpl": "SELECT player_id, player_name, sport, {period}, {agg}(ranking_points) AS {col} FROM sports_rankings_pts GROUP BY player_id, player_name, sport, {period}"},
    {"job": "ingest_venues", "inputs": ["source/synthetic-venues"], "outputs": [
        "venues_capacity_exact", "venues_capacity_thousands",
    ], "sql_tmpl": "SELECT venue_id, venue_name, city, country, {cap_col} FROM synthetic_venues"},
    {"job": "aggregate_venue_attendance", "inputs": ["venues_attendance_match"], "outputs": [
        "venues_attendance_season", "venues_attendance_annual",
    ], "sql_tmpl": "SELECT venue_id, venue_name, {period}, {agg}(attendance) AS {col} FROM venues_attendance_match GROUP BY venue_id, venue_name, {period}"},
    {"job": "ingest_venue_matches", "inputs": ["source/synthetic-events"], "outputs": ["venues_attendance_match"],
     "sql_tmpl": "SELECT CONCAT('M', LPAD(ROW_NUMBER() OVER (), 5, '0')) AS match_id, venue_id, venue_name, event_date, attendance FROM synthetic_events"},
    {"job": "ingest_athletes", "inputs": ["source/synthetic-athletes"], "outputs": [
        "athletes_nat_full", "athletes_nat_iso2", "athletes_nat_iso3",
    ], "sql_tmpl": "SELECT athlete_id, athlete_name, sport, {nat_col} FROM synthetic_athletes"},
    {"job": "ingest_employees_nationality", "inputs": ["source/synthetic-employees"], "outputs": [
        "employees_nationality_full", "employees_nationality_iso2",
    ], "sql_tmpl": "SELECT employee_id, full_name, department, {nat_col} FROM synthetic_employees"},
    {"job": "ingest_survey_gender", "inputs": ["source/synthetic-survey"], "outputs": [
        "survey_gender_full", "survey_gender_abbr", "survey_gender_binary",
    ], "sql_tmpl": "SELECT respondent_id, age_group, country, {gender_col} FROM synthetic_survey"},
    {"job": "ingest_employees_gender", "inputs": ["source/synthetic-employees"], "outputs": [
        "employees_gender_full", "employees_gender_code",
    ], "sql_tmpl": "SELECT employee_id, department, {gender_col} FROM synthetic_employees"},
    {"job": "ingest_persons", "inputs": ["source/synthetic-persons"], "outputs": [
        "persons_fullname", "persons_lastfirst", "persons_initials",
    ], "sql_tmpl": "SELECT person_id, {name_col} AS {col_alias}, department, company FROM synthetic_persons"},
    {"job": "ingest_employees_names", "inputs": ["source/synthetic-employees"], "outputs": [
        "employees_fullname", "employees_lastfirst",
    ], "sql_tmpl": "SELECT employee_id, {name_col} AS {col_alias}, department, company FROM synthetic_employees"},
    {"job": "ingest_authors", "inputs": ["source/synthetic-authors"], "outputs": [
        "authors_fullname", "authors_lastfirst",
    ], "sql_tmpl": "SELECT author_id, {name_col} AS {col_alias}, genre, country FROM synthetic_authors"},
    {"job": "ingest_movies_directors", "inputs": ["source/synthetic-movies"], "outputs": [
        "movies_directors", "movies_director_abbr", "movies_director_lastfirst",
    ], "sql_tmpl": "SELECT movie_id, name, {dir_col} AS {col_alias}, year, language, genre FROM synthetic_movies"},
    {"job": "ingest_movies_cast", "inputs": ["source/synthetic-movies"], "outputs": [
        "movies_cast", "movies_cast_abbr",
    ], "sql_tmpl": "SELECT movie_id, name, year, {actor_col} FROM synthetic_movies_cast"},
    {"job": "aggregate_cast_summary", "inputs": ["movies_cast"], "outputs": ["cast_summary"],
     "sql_tmpl": "SELECT actor, COUNT(DISTINCT name) AS total_movies FROM movies_cast GROUP BY actor"},
    {"job": "ingest_series_directors", "inputs": ["source/synthetic-series"], "outputs": [
        "series_directors", "series_director_abbr",
    ], "sql_tmpl": "SELECT series_id, name, {dir_col} AS {col_alias}, year, language, genre FROM synthetic_series"},
    {"job": "ingest_salary_bands", "inputs": ["source/synthetic-compensation"], "outputs": [
        "salary_bands_usd", "salary_bands_eur",
    ], "sql_tmpl": "SELECT department, grade, level, {min_col}, {max_col}, {mid_col} FROM synthetic_compensation"},
    {"job": "ingest_age_distribution", "inputs": ["source/synthetic-population"], "outputs": [
        "age_dist_5yr", "age_dist_10yr",
    ], "sql_tmpl": "SELECT age_range, population_count, ROUND(population_count * 100.0 / SUM(population_count) OVER (), 2) AS pct_total FROM synthetic_population GROUP BY {bin_size}, age_range"},
    {"job": "ingest_price_tiers", "inputs": ["source/synthetic-products"], "outputs": [
        "product_price_tiers_usd", "product_price_tiers_eur",
    ], "sql_tmpl": "SELECT product_id, category, {tier_col} AS {col_alias}, brand FROM synthetic_products"},
    {"job": "ingest_athletes_ioc", "inputs": ["source/synthetic-athletes"], "outputs": ["athletes_nat_ioc"],
     "sql_tmpl": "SELECT athlete_id, athlete_name, sport, IOC_MAP[nationality] AS nationality_ioc FROM synthetic_athletes"},
    {"job": "aggregate_products_temporal_eur", "inputs": ["products_sales_eur"], "outputs": [
        "products_monthly_eur", "products_quarterly_eur",
    ], "sql_tmpl": "SELECT product_id, product_name, brand, category, {period} AS report_{grain}, ROUND(annual_sales_eur / {divisor}, 0) AS {col} FROM products_sales_eur"},
    {"job": "ingest_employee_salaries", "inputs": ["source/synthetic-employees"], "outputs": [
        "employee_salary_usd", "employee_salary_eur",
    ], "sql_tmpl": "SELECT employee_id, full_name, department, grade, level, {annual_col}, {monthly_col} FROM synthetic_employees"},
    {"job": "ingest_city_climate", "inputs": ["source/synthetic-weather"], "outputs": [
        "city_climate_celsius", "city_climate_fahrenheit",
    ], "sql_tmpl": "SELECT city, country, month, {temp_col} FROM synthetic_weather"},
    {"job": "aggregate_city_climate_annual", "inputs": ["city_climate_celsius"], "outputs": [
        "city_climate_annual_celsius", "city_climate_annual_fahrenheit",
    ], "sql_tmpl": "SELECT city, country, AVG({temp_col}) AS {annual_col} FROM city_climate_celsius GROUP BY city, country"},
]


def build_lineage_events() -> list[dict]:
    events = []
    run_id = 1
    event_time_base = "2024-06-01T09:00:00Z"
    for tmpl in LINEAGE_TEMPLATES:
        for out_table in tmpl["outputs"]:
            events.append({
                "eventType": "COMPLETE",
                "eventTime": event_time_base,
                "run": {"runId": f"dag-run-{run_id:04d}"},
                "job": {"namespace": "lakehouse", "name": tmpl["job"]},
                "inputs": [{"namespace": "source" if "/" in inp else "lakehouse", "name": inp} for inp in tmpl["inputs"]],
                "outputs": [{
                    "namespace": "lakehouse",
                    "name": out_table,
                    "facets": {"sql": {"query": tmpl["sql_tmpl"]}},
                }],
            })
            run_id += 1
    return events


def build_manifest(table: str, df: pd.DataFrame) -> dict:
    cols: dict[str, dict] = {}
    for col in df.columns:
        series = df[col].dropna()
        entry: dict = {
            "row_count": len(df),
            "null_count": int(df[col].isna().sum()),
            "distinct_count": int(series.nunique()),
        }
        sample = series.sample(min(6, len(series)), random_state=42).tolist()
        entry["sample_values"] = [str(v) for v in sample]
        if pd.api.types.is_numeric_dtype(series) and len(series) > 0:
            entry["min"] = round(float(series.min()), 4)
            entry["max"] = round(float(series.max()), 4)
            entry["mean"] = round(float(series.mean()), 4)
            entry["std"] = round(float(series.std()), 4)
        cols[col] = entry
    return {"table": table, "columns": cols}


def build_pairs() -> list[dict]:
    pairs = []
    for i, tmpl in enumerate(PAIR_TEMPLATES, 1):
        pairs.append({
            "id": f"P{i:04d}",
            "table_a": tmpl["a"][0],
            "col_a": tmpl["a"][1],
            "table_b": tmpl["b"][0],
            "col_b": tmpl["b"][1],
            "label": tmpl["label"],
        })
    return pairs


def print_distribution(pairs: list[dict]) -> None:
    from collections import Counter
    dist = Counter(p["label"] for p in pairs)
    total = len(pairs)
    print(f"\nTotal pairs: {total}")
    for label in ["TYPE1_MEASURE", "TYPE2_GRANULARITY", "NO_CONFLICT_DUPLICATE", "NO_CONFLICT_DIFF_ENTITY"]:
        n = dist.get(label, 0)
        print(f"  {label}: {n} ({n / total * 100:.1f}%)")


# Main guard
if __name__ == "__main__":
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    table_files = [f.replace(".csv", "") for f in os.listdir(TABLES_DIR) if f.endswith(".csv")]
    print(f"Found {len(table_files)} tables in {TABLES_DIR}")

    pairs = build_pairs()
    missing = set()
    for p in pairs:
        for t in [p["table_a"], p["table_b"]]:
            if t not in table_files:
                missing.add(t)
    if missing:
        print(f"WARNING: missing tables: {sorted(missing)}")

    pairs_path = os.path.join(SCRIPT_DIR, "..", "dataset", "pairs.json")
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print_distribution(pairs)
    print(f"\npairs.json written ({len(pairs)} pairs)")

    ddl_path = os.path.join(META_DIR, "ddl.sql")
    with open(ddl_path, "w") as f:
        for t in sorted(DDL_TEMPLATES):
            f.write(DDL_TEMPLATES[t] + "\n\n")
    print(f"ddl.sql written ({len(DDL_TEMPLATES)} tables)")

    lineage_events = build_lineage_events()
    lineage_path = os.path.join(META_DIR, "lineage.json")
    with open(lineage_path, "w") as f:
        json.dump(lineage_events, f, indent=2)
    print(f"lineage.json written ({len(lineage_events)} events)")

    manifest_count = 0
    for table in table_files:
        csv_path = os.path.join(TABLES_DIR, f"{table}.csv")
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            manifest = build_manifest(table, df)
            mfst_path = os.path.join(MANIFESTS_DIR, f"{table}.json")
            with open(mfst_path, "w") as f:
                json.dump(manifest, f, indent=2)
            manifest_count += 1
        except Exception as e:
            print(f"  WARNING: could not process {table}: {e}")
    print(f"manifests written ({manifest_count} files)")

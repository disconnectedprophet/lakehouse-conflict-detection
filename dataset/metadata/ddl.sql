CREATE TABLE age_dist_10yr (
    age_range STRING COMMENT 'Ten-year age group (e.g. 20-29)',
    population_count BIGINT COMMENT 'Population count in age group',
    pct_total DOUBLE COMMENT 'Percentage of total population'
)
COMMENT 'Population age distribution in 10-year bins'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' granularity='10yr' grain='per-age-bin');

CREATE TABLE age_dist_5yr (
    age_range STRING COMMENT 'Five-year age group (e.g. 25-29)',
    population_count BIGINT COMMENT 'Population count in age group',
    pct_total DOUBLE COMMENT 'Percentage of total population'
)
COMMENT 'Population age distribution in 5-year bins'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' granularity='5yr' grain='per-age-bin');

CREATE TABLE athletes_nat_full (
    athlete_id STRING COMMENT 'Unique athlete identifier',
    athlete_name STRING COMMENT 'Athlete full name',
    sport STRING COMMENT 'Sport discipline',
    nationality STRING COMMENT 'Nationality full country name'
)
COMMENT 'Athlete roster with full nationality names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-country-name' grain='per-athlete');

CREATE TABLE athletes_nat_ioc (
    athlete_id STRING COMMENT 'Unique athlete identifier',
    athlete_name STRING COMMENT 'Athlete full name',
    sport STRING COMMENT 'Sport discipline',
    nationality_ioc STRING COMMENT 'IOC 3-letter nationality code (differs from ISO3 for 8 countries: GER/NED/SUI/GRE/DEN/CRO/POR/RSA)'
)
COMMENT 'Athlete roster with IOC nationality codes used in Olympic competition'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='IOC-3letter' grain='per-athlete');

CREATE TABLE athletes_nat_iso2 (
    athlete_id STRING COMMENT 'Athlete identifier',
    athlete_name STRING COMMENT 'Athlete name',
    sport STRING COMMENT 'Sport',
    nationality_iso2 STRING COMMENT 'ISO 3166-1 alpha-2 nationality code (e.g. ES)'
)
COMMENT 'Athlete roster with ISO 3166-1 alpha-2 nationality codes'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='ISO-3166-1-alpha-2' grain='per-athlete');

CREATE TABLE athletes_nat_iso3 (
    athlete_id STRING COMMENT 'Athlete identifier',
    athlete_name STRING COMMENT 'Athlete name',
    sport STRING COMMENT 'Sport',
    nationality_iso3 STRING COMMENT 'ISO 3166-1 alpha-3 nationality code (e.g. ESP)'
)
COMMENT 'Athlete roster with ISO 3166-1 alpha-3 nationality codes'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='ISO-3166-1-alpha-3' grain='per-athlete');

CREATE TABLE authors_fullname (
    author_id STRING COMMENT 'Unique author identifier',
    full_name STRING COMMENT 'Author full name',
    genre STRING COMMENT 'Primary writing genre',
    country STRING COMMENT 'Country of origin'
)
COMMENT 'Author directory with full names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-author');

CREATE TABLE authors_lastfirst (
    author_id STRING COMMENT 'Unique author identifier',
    last_first STRING COMMENT 'Author name in Lastname, Firstname format',
    genre STRING COMMENT 'Primary writing genre',
    country STRING COMMENT 'Country of origin'
)
COMMENT 'Author directory with last-first name format'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='last-first' grain='per-author');

CREATE TABLE brand_category_summary (
    brand STRING COMMENT 'Brand abbreviation',
    brand_full STRING COMMENT 'Full legal brand name',
    total_products INT COMMENT 'Number of products in dataset',
    total_sales_usd BIGINT COMMENT 'Total sales across all products in USD',
    avg_sales_usd BIGINT COMMENT 'Average sales per product in USD'
)
COMMENT 'Per-brand aggregate sales summary'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' grain='per-brand');

CREATE TABLE cast_summary (
    actor STRING COMMENT 'Actor full name',
    total_movies INT COMMENT 'Number of movies the actor appeared in'
)
COMMENT 'Per-actor movie count summary'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' grain='per-actor');

CREATE TABLE city_climate_annual_celsius (
    city STRING COMMENT 'City name',
    country STRING COMMENT 'Country',
    annual_avg_celsius DOUBLE COMMENT 'Annual average temperature in degrees Celsius'
)
COMMENT 'Annual average city temperatures in Celsius'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' unit='celsius' grain='per-city-annual');

CREATE TABLE city_climate_annual_fahrenheit (
    city STRING COMMENT 'City name',
    country STRING COMMENT 'Country',
    annual_avg_fahrenheit DOUBLE COMMENT 'Annual average temperature in degrees Fahrenheit'
)
COMMENT 'Annual average city temperatures in Fahrenheit'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' unit='fahrenheit' grain='per-city-annual');

CREATE TABLE city_climate_celsius (
    city STRING COMMENT 'City name',
    country STRING COMMENT 'Country',
    month INT COMMENT 'Month number (1-12)',
    avg_temp_celsius DOUBLE COMMENT 'Average monthly temperature in degrees Celsius'
)
COMMENT 'Monthly average city temperatures in Celsius'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' unit='celsius' grain='per-city-month');

CREATE TABLE city_climate_fahrenheit (
    city STRING COMMENT 'City name',
    country STRING COMMENT 'Country',
    month INT COMMENT 'Month number (1-12)',
    avg_temp_fahrenheit DOUBLE COMMENT 'Average monthly temperature in degrees Fahrenheit'
)
COMMENT 'Monthly average city temperatures in Fahrenheit'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' unit='fahrenheit' grain='per-city-month');

CREATE TABLE employee_salary_eur (
    employee_id STRING COMMENT 'Unique employee identifier',
    full_name STRING COMMENT 'Employee full name',
    department STRING COMMENT 'Department',
    grade STRING COMMENT 'Compensation grade (e.g. IC3)',
    level STRING COMMENT 'Level label (e.g. Senior)',
    annual_salary_eur INT COMMENT 'Annual base salary in EUR',
    monthly_salary_eur INT COMMENT 'Monthly base salary in EUR (annual / 12)'
)
COMMENT 'Individual employee salaries in EUR with annual and monthly grain'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='EUR' grain='per-employee');

CREATE TABLE employee_salary_usd (
    employee_id STRING COMMENT 'Unique employee identifier',
    full_name STRING COMMENT 'Employee full name',
    department STRING COMMENT 'Department',
    grade STRING COMMENT 'Compensation grade (e.g. IC3)',
    level STRING COMMENT 'Level label (e.g. Senior)',
    annual_salary_usd INT COMMENT 'Annual base salary in USD',
    monthly_salary_usd INT COMMENT 'Monthly base salary in USD (annual / 12)'
)
COMMENT 'Individual employee salaries in USD with annual and monthly grain'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='USD' grain='per-employee');

CREATE TABLE employees_fullname (
    employee_id STRING COMMENT 'Unique employee identifier',
    full_name STRING COMMENT 'Full name (Firstname Lastname)',
    department STRING COMMENT 'Department',
    company STRING COMMENT 'Company'
)
COMMENT 'Employee directory with full names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-employee');

CREATE TABLE employees_gender_code (
    employee_id STRING COMMENT 'Unique employee identifier',
    department STRING COMMENT 'Department',
    gender_code INT COMMENT 'Gender numeric code (0=Female 1=Male 2=Non-binary)'
)
COMMENT 'Employee gender with numeric codes'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='numeric' grain='per-employee');

CREATE TABLE employees_gender_full (
    employee_id STRING COMMENT 'Unique employee identifier',
    department STRING COMMENT 'Department',
    gender STRING COMMENT 'Gender full text'
)
COMMENT 'Employee gender with full-text values'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-text' grain='per-employee');

CREATE TABLE employees_lastfirst (
    employee_id STRING COMMENT 'Unique employee identifier',
    last_first STRING COMMENT 'Name in Lastname, Firstname format',
    department STRING COMMENT 'Department',
    company STRING COMMENT 'Company'
)
COMMENT 'Employee directory with last-first name format'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='last-first' grain='per-employee');

CREATE TABLE employees_nationality_full (
    employee_id STRING COMMENT 'Unique employee identifier',
    full_name STRING COMMENT 'Employee full name',
    department STRING COMMENT 'Department',
    nationality STRING COMMENT 'Nationality full country name'
)
COMMENT 'Employee roster with full nationality names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-country-name' grain='per-employee');

CREATE TABLE employees_nationality_iso2 (
    employee_id STRING COMMENT 'Unique employee identifier',
    full_name STRING COMMENT 'Employee full name',
    department STRING COMMENT 'Department',
    nationality_iso2 STRING COMMENT 'ISO 3166-1 alpha-2 nationality code'
)
COMMENT 'Employee roster with ISO 3166-1 alpha-2 nationality codes'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='ISO-3166-1-alpha-2' grain='per-employee');

CREATE TABLE movies_cast (
    movie_id STRING COMMENT 'Movie identifier',
    name STRING COMMENT 'Movie title',
    year INT COMMENT 'Release year',
    actor STRING COMMENT 'Actor full name'
)
COMMENT 'Movie cast with actor full names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-movie-actor');

CREATE TABLE movies_cast_abbr (
    movie_id STRING COMMENT 'Movie identifier',
    name STRING COMMENT 'Movie title',
    year INT COMMENT 'Release year',
    actor_abbr STRING COMMENT 'Actor name abbreviated (F. Lastname)'
)
COMMENT 'Movie cast with abbreviated actor names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-movie-actor');

CREATE TABLE movies_director_abbr (
    movie_id STRING COMMENT 'Unique movie identifier',
    name STRING COMMENT 'Movie title',
    director_abbr STRING COMMENT 'Director name abbreviated (F. Lastname)',
    year INT COMMENT 'Release year',
    language STRING COMMENT 'Original language code',
    genre STRING COMMENT 'Primary genre'
)
COMMENT 'Movie catalog with abbreviated director names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-movie');

CREATE TABLE movies_director_lastfirst (
    movie_id STRING COMMENT 'Unique movie identifier',
    name STRING COMMENT 'Movie title',
    director_lastfirst STRING COMMENT 'Director name in Lastname, Firstname format',
    year INT COMMENT 'Release year',
    language STRING COMMENT 'Original language code',
    genre STRING COMMENT 'Primary genre'
)
COMMENT 'Movie catalog with last-first director names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='last-first' grain='per-movie');

CREATE TABLE movies_directors (
    movie_id STRING COMMENT 'Unique movie identifier',
    name STRING COMMENT 'Movie title',
    director STRING COMMENT 'Director full name',
    year INT COMMENT 'Release year',
    language STRING COMMENT 'Original language code',
    genre STRING COMMENT 'Primary genre'
)
COMMENT 'Movie catalog with director full names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-movie');

CREATE TABLE persons_fullname (
    person_id STRING COMMENT 'Unique person identifier',
    full_name STRING COMMENT 'Full name (Firstname Lastname)',
    department STRING COMMENT 'Department',
    company STRING COMMENT 'Company'
)
COMMENT 'Person directory with full names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-person');

CREATE TABLE persons_initials (
    person_id STRING COMMENT 'Unique person identifier',
    initials_name STRING COMMENT 'Name with abbreviated first name (F. Lastname)',
    department STRING COMMENT 'Department',
    company STRING COMMENT 'Company'
)
COMMENT 'Person directory with abbreviated first name'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-person');

CREATE TABLE persons_lastfirst (
    person_id STRING COMMENT 'Unique person identifier',
    last_first STRING COMMENT 'Name in Lastname, Firstname format',
    department STRING COMMENT 'Department',
    company STRING COMMENT 'Company'
)
COMMENT 'Person directory with last-first name format'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='last-first' grain='per-person');

CREATE TABLE product_price_tiers_eur (
    product_id STRING COMMENT 'Unique product identifier',
    category STRING COMMENT 'Product category',
    price_tier_eur STRING COMMENT 'Price tier label with EUR range (e.g. Mid-range (91-228))',
    brand STRING COMMENT 'Brand name'
)
COMMENT 'Product price tier classification in EUR ranges'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='EUR' grain='per-product');

CREATE TABLE product_price_tiers_usd (
    product_id STRING COMMENT 'Unique product identifier',
    category STRING COMMENT 'Product category',
    price_tier STRING COMMENT 'Price tier label with USD range (e.g. Mid-range (100-250))',
    brand STRING COMMENT 'Brand name'
)
COMMENT 'Product price tier classification in USD ranges'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='USD' grain='per-product');

CREATE TABLE products_brand_abbr (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand_abbr STRING COMMENT 'Brand abbreviation or common short name',
    category STRING COMMENT 'Product category',
    annual_sales_usd BIGINT COMMENT 'Annual sales in USD'
)
COMMENT 'Product catalog with abbreviated brand names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-product');

CREATE TABLE products_brand_full (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand_full STRING COMMENT 'Full legal brand name',
    category STRING COMMENT 'Product category',
    annual_sales_usd BIGINT COMMENT 'Annual sales in USD'
)
COMMENT 'Product catalog with full legal brand names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-legal' grain='per-product');

CREATE TABLE products_monthly_eur (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    report_month STRING COMMENT 'Reporting month (YYYY-MM)',
    monthly_sales_eur BIGINT COMMENT 'Monthly sales in euros'
)
COMMENT 'Monthly product sales in EUR'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='EUR' grain='per-product-monthly');

CREATE TABLE products_monthly_usd (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    report_month STRING COMMENT 'Reporting month (YYYY-MM)',
    monthly_sales_usd BIGINT COMMENT 'Monthly sales in USD'
)
COMMENT 'Monthly product sales in USD'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='USD' grain='per-product-monthly');

CREATE TABLE products_quarterly_eur (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    report_quarter STRING COMMENT 'Reporting quarter (Q1-2023)',
    quarterly_sales_eur BIGINT COMMENT 'Quarterly sales in euros'
)
COMMENT 'Quarterly product sales in EUR'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='EUR' grain='per-product-quarterly');

CREATE TABLE products_quarterly_usd (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    report_quarter STRING COMMENT 'Reporting quarter (Q1-2023)',
    quarterly_sales_usd BIGINT COMMENT 'Quarterly sales in USD'
)
COMMENT 'Quarterly product sales in USD'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='USD' grain='per-product-quarterly');

CREATE TABLE products_sales_chf (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    annual_sales_chf BIGINT COMMENT 'Annual sales in Swiss francs',
    units_sold BIGINT COMMENT 'Total units sold'
)
COMMENT 'Annual product sales in CHF'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='CHF' grain='per-product-annual');

CREATE TABLE products_sales_eur (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    annual_sales_eur BIGINT COMMENT 'Annual sales in euros',
    units_sold BIGINT COMMENT 'Total units sold'
)
COMMENT 'Annual product sales in EUR'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='EUR' grain='per-product-annual');

CREATE TABLE products_sales_gbp (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    annual_sales_gbp BIGINT COMMENT 'Annual sales in British pounds',
    units_sold BIGINT COMMENT 'Total units sold'
)
COMMENT 'Annual product sales in GBP'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='GBP' grain='per-product-annual');

CREATE TABLE products_sales_jpy (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    annual_sales_jpy BIGINT COMMENT 'Annual sales in Japanese yen',
    units_sold BIGINT COMMENT 'Total units sold'
)
COMMENT 'Annual product sales in JPY'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='JPY' grain='per-product-annual');

CREATE TABLE products_sales_usd (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    annual_sales_usd BIGINT COMMENT 'Annual sales in US dollars',
    units_sold BIGINT COMMENT 'Total units sold'
)
COMMENT 'Annual product sales in USD'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='USD' grain='per-product-annual');

CREATE TABLE products_weekly_usd (
    product_id STRING COMMENT 'Unique product identifier',
    product_name STRING COMMENT 'Product name',
    brand STRING COMMENT 'Brand name',
    category STRING COMMENT 'Product category',
    report_week STRING COMMENT 'ISO week (2023-W01)',
    weekly_sales_usd BIGINT COMMENT 'Weekly sales in USD'
)
COMMENT 'Weekly product sales in USD'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='USD' grain='per-product-weekly');

CREATE TABLE retailers_brand_abbr (
    store_id STRING COMMENT 'Unique store identifier',
    store_name STRING COMMENT 'Store name',
    brand_abbr STRING COMMENT 'Retailer brand abbreviation',
    country STRING COMMENT 'Country of operation',
    annual_revenue_eur BIGINT COMMENT 'Annual store revenue in EUR'
)
COMMENT 'Retail store catalog with abbreviated brand names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-store');

CREATE TABLE retailers_brand_full (
    store_id STRING COMMENT 'Unique store identifier',
    store_name STRING COMMENT 'Store name',
    brand_full STRING COMMENT 'Retailer full legal brand name',
    country STRING COMMENT 'Country of operation',
    annual_revenue_eur BIGINT COMMENT 'Annual store revenue in EUR'
)
COMMENT 'Retail store catalog with full legal brand names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-legal' grain='per-store');

CREATE TABLE salary_bands_eur (
    department STRING COMMENT 'Department name',
    grade STRING COMMENT 'Compensation grade (e.g. IC3)',
    level STRING COMMENT 'Level label (e.g. Senior)',
    min_salary_eur INT COMMENT 'Minimum salary for grade in EUR',
    max_salary_eur INT COMMENT 'Maximum salary for grade in EUR',
    midpoint_eur INT COMMENT 'Midpoint salary for grade in EUR'
)
COMMENT 'Salary band ranges per grade in EUR'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='EUR' grain='per-department-grade');

CREATE TABLE salary_bands_usd (
    department STRING COMMENT 'Department name',
    grade STRING COMMENT 'Compensation grade (e.g. IC3)',
    level STRING COMMENT 'Level label (e.g. Senior)',
    min_salary_usd INT COMMENT 'Minimum salary for grade in USD',
    max_salary_usd INT COMMENT 'Maximum salary for grade in USD',
    midpoint_usd INT COMMENT 'Midpoint salary for grade in USD'
)
COMMENT 'Salary band ranges per grade in USD'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' currency='USD' grain='per-department-grade');

CREATE TABLE series_director_abbr (
    series_id STRING COMMENT 'Unique series identifier',
    name STRING COMMENT 'Series title',
    director_abbr STRING COMMENT 'Director name abbreviated (F. Lastname)',
    year INT COMMENT 'Release year',
    language STRING COMMENT 'Original language code',
    genre STRING COMMENT 'Primary genre'
)
COMMENT 'TV series catalog with abbreviated director names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-series');

CREATE TABLE series_directors (
    series_id STRING COMMENT 'Unique series identifier',
    name STRING COMMENT 'Series title',
    director STRING COMMENT 'Director full name',
    year INT COMMENT 'Release year',
    language STRING COMMENT 'Original language code',
    genre STRING COMMENT 'Primary genre'
)
COMMENT 'TV series catalog with director full names'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-name' grain='per-series');

CREATE TABLE sports_rankings_career (
    player_id STRING COMMENT 'Unique player identifier',
    player_name STRING COMMENT 'Player full name',
    sport STRING COMMENT 'Sport discipline',
    career_span STRING COMMENT 'Career period (2015-2023)',
    career_total_points INT COMMENT 'Career total points accumulated'
)
COMMENT 'Career total player ranking points'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' grain='per-player-career');

CREATE TABLE sports_rankings_elo (
    player_id STRING COMMENT 'Unique player identifier',
    player_name STRING COMMENT 'Player full name',
    nationality STRING COMMENT 'Nationality',
    sport STRING COMMENT 'Sport discipline',
    elo_rating INT COMMENT 'ELO-style rating (baseline 1200)',
    rank_elo INT COMMENT 'Rank by ELO rating'
)
COMMENT 'Player rankings using ELO rating system'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' scoring='elo' grain='per-player');

CREATE TABLE sports_rankings_monthly (
    player_id STRING COMMENT 'Unique player identifier',
    player_name STRING COMMENT 'Player full name',
    sport STRING COMMENT 'Sport discipline',
    report_month STRING COMMENT 'Reporting month (YYYY-MM)',
    monthly_avg_points INT COMMENT 'Average monthly points'
)
COMMENT 'Monthly average player ranking points'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' grain='per-player-monthly');

CREATE TABLE sports_rankings_norm (
    player_id STRING COMMENT 'Unique player identifier',
    player_name STRING COMMENT 'Player full name',
    nationality STRING COMMENT 'Nationality',
    sport STRING COMMENT 'Sport discipline',
    points_norm DOUBLE COMMENT 'Normalised ranking score (0-100 scale)',
    rank_percentile DOUBLE COMMENT 'Rank percentile (1.0 = top)'
)
COMMENT 'Player rankings normalised to 0-100 scale'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' scoring='normalised' grain='per-player');

CREATE TABLE sports_rankings_pts (
    player_id STRING COMMENT 'Unique player identifier',
    player_name STRING COMMENT 'Player full name',
    nationality STRING COMMENT 'Nationality full country name',
    sport STRING COMMENT 'Sport discipline',
    ranking_points INT COMMENT 'Raw ranking points',
    world_rank INT COMMENT 'World ranking position'
)
COMMENT 'Player rankings in raw points'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' scoring='raw-points' grain='per-player');

CREATE TABLE sports_rankings_season (
    player_id STRING COMMENT 'Unique player identifier',
    player_name STRING COMMENT 'Player full name',
    sport STRING COMMENT 'Sport discipline',
    season STRING COMMENT 'Season identifier (2022-2023)',
    season_avg_points INT COMMENT 'Season average points'
)
COMMENT 'Season average player ranking points'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' grain='per-player-season');

CREATE TABLE survey_gender_abbr (
    respondent_id STRING COMMENT 'Unique respondent identifier',
    age_group STRING COMMENT 'Age group range',
    country STRING COMMENT 'Country',
    gender_abbr STRING COMMENT 'Gender abbreviation (M / F / NB)'
)
COMMENT 'Survey responses with abbreviated gender codes'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='abbreviated' grain='per-respondent');

CREATE TABLE survey_gender_binary (
    respondent_id STRING COMMENT 'Unique respondent identifier',
    age_group STRING COMMENT 'Age group range',
    country STRING COMMENT 'Country',
    gender_code INT COMMENT 'Gender numeric code (0=Female 1=Male 2=Non-binary)'
)
COMMENT 'Survey responses with numeric gender codes'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='numeric' grain='per-respondent');

CREATE TABLE survey_gender_full (
    respondent_id STRING COMMENT 'Unique respondent identifier',
    age_group STRING COMMENT 'Age group range (e.g. 25-34)',
    country STRING COMMENT 'Country of respondent',
    gender STRING COMMENT 'Gender full text (Male / Female / Non-binary)'
)
COMMENT 'Survey responses with full-text gender values'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' encoding='full-text' grain='per-respondent');

CREATE TABLE venues_attendance_annual (
    venue_id STRING COMMENT 'Venue identifier',
    venue_name STRING COMMENT 'Venue name',
    year INT COMMENT 'Calendar year',
    total_annual_attendance INT COMMENT 'Total attendance across all matches in the year',
    match_count INT COMMENT 'Number of matches held'
)
COMMENT 'Annual total attendance per venue'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' grain='per-venue-annual');

CREATE TABLE venues_attendance_match (
    match_id STRING COMMENT 'Unique match identifier',
    venue_id STRING COMMENT 'Venue identifier',
    venue_name STRING COMMENT 'Venue name',
    event_date STRING COMMENT 'Event date (YYYY-MM-DD)',
    attendance INT COMMENT 'Per-match attendance count'
)
COMMENT 'Per-match attendance records'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' grain='per-match');

CREATE TABLE venues_attendance_season (
    venue_id STRING COMMENT 'Venue identifier',
    venue_name STRING COMMENT 'Venue name',
    season STRING COMMENT 'Season identifier',
    avg_season_attendance INT COMMENT 'Average attendance per match in the season',
    match_count INT COMMENT 'Number of matches in the season'
)
COMMENT 'Season average attendance per venue'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' grain='per-venue-season');

CREATE TABLE venues_capacity_exact (
    venue_id STRING COMMENT 'Unique venue identifier',
    venue_name STRING COMMENT 'Venue name',
    city STRING COMMENT 'City',
    country STRING COMMENT 'Country',
    capacity_seats INT COMMENT 'Seating capacity in individual seats',
    standing_capacity INT COMMENT 'Standing area capacity'
)
COMMENT 'Venue capacity in exact seat count'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' unit='seats' grain='per-venue');

CREATE TABLE venues_capacity_thousands (
    venue_id STRING COMMENT 'Unique venue identifier',
    venue_name STRING COMMENT 'Venue name',
    city STRING COMMENT 'City',
    country STRING COMMENT 'Country',
    capacity_thousands DOUBLE COMMENT 'Seating capacity in thousands of seats',
    standing_thousands DOUBLE COMMENT 'Standing capacity in thousands'
)
COMMENT 'Venue capacity in thousands of seats'
STORED AS PARQUET
TBLPROPERTIES (source='synthetic' unit='thousands' grain='per-venue');


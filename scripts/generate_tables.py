"""
Generates all synthetic CSV tables for the datalake conflict-detection dataset.

Targets the 10 Sherlock entity types with recall below 0.75:
  ranking (0.312), sales (0.528), director (0.547), person (0.618),
  brand (0.671), nationality (0.691), gender (0.721), capacity (0.721),
  range (0.759), name (0.759).

Tables written to ../dataset/tables/.

Run via:
  python3 generate_tables.py
"""

# Libraries
import math
import os
import random
import numpy as np
import pandas as pd
random.seed(42)
np.random.seed(42)

# Configuration variables
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, "..", "dataset", "tables")

USD_EUR = 0.91
USD_GBP = 0.79
USD_JPY = 149.5
USD_CHF = 0.88

IOC_MAP: dict[str, str] = {
    "Spain": "ESP", "France": "FRA", "Germany": "GER", "Italy": "ITA",
    "United Kingdom": "GBR", "United States": "USA", "Brazil": "BRA",
    "Argentina": "ARG", "Australia": "AUS", "Japan": "JPN", "Canada": "CAN",
    "Netherlands": "NED", "Serbia": "SRB", "Czech Republic": "CZE",
    "Croatia": "CRO", "Poland": "POL", "Portugal": "POR", "Sweden": "SWE",
    "Switzerland": "SUI", "Norway": "NOR", "Belgium": "BEL", "Austria": "AUT",
    "Greece": "GRE", "Denmark": "DEN", "South Korea": "KOR", "Russia": "RUS",
    "Mexico": "MEX", "China": "CHN", "India": "IND", "South Africa": "RSA",
}

CITY_LATITUDE: dict[str, float] = {
    "Oslo": 59.9, "Helsinki": 60.2, "Stockholm": 59.3, "Gothenburg": 57.7,
    "Copenhagen": 55.7, "Aarhus": 56.2, "Hamburg": 53.6, "Dublin": 53.3,
    "Amsterdam": 52.4, "Rotterdam": 51.9, "Brussels": 50.8, "Ghent": 51.0,
    "Warsaw": 52.2, "Krakow": 50.1, "Berlin": 52.5, "Prague": 50.1,
    "Vienna": 48.2, "Salzburg": 47.8, "Zurich": 47.4, "Budapest": 47.5,
    "Lyon": 45.8, "Turin": 45.1, "Paris": 48.9, "London": 51.5,
    "Porto": 41.2, "Lisbon": 38.7, "Athens": 37.9, "Madrid": 40.4,
    "Barcelona": 41.4, "Milan": 45.5,
}

NAT_MAP: dict[str, tuple[str, str]] = {
    "Spain": ("ESP", "ES"), "France": ("FRA", "FR"), "Germany": ("DEU", "DE"),
    "Italy": ("ITA", "IT"), "United Kingdom": ("GBR", "GB"), "United States": ("USA", "US"),
    "Brazil": ("BRA", "BR"), "Argentina": ("ARG", "AR"), "Australia": ("AUS", "AU"),
    "Japan": ("JPN", "JP"), "Canada": ("CAN", "CA"), "Netherlands": ("NLD", "NL"),
    "Serbia": ("SRB", "RS"), "Czech Republic": ("CZE", "CZ"), "Croatia": ("HRV", "HR"),
    "Poland": ("POL", "PL"), "Portugal": ("PRT", "PT"), "Sweden": ("SWE", "SE"),
    "Switzerland": ("CHE", "CH"), "Norway": ("NOR", "NO"), "Belgium": ("BEL", "BE"),
    "Austria": ("AUT", "AT"), "Greece": ("GRC", "GR"), "Denmark": ("DNK", "DK"),
    "South Korea": ("KOR", "KR"), "Russia": ("RUS", "RU"), "Mexico": ("MEX", "MX"),
    "China": ("CHN", "CN"), "India": ("IND", "IN"), "South Africa": ("ZAF", "ZA"),
}

BRANDS: list[str] = [
    "Samsung", "Apple", "Sony", "LG", "Philips", "Bosch", "Nestlé", "Unilever",
    "P&G", "Nike", "Adidas", "Puma", "IKEA", "H&M", "Zara",
    "Panasonic", "Siemens", "3M", "Colgate", "Heinz",
]
BRAND_MAP: dict[str, str] = {
    "Samsung": "Samsung Electronics Co., Ltd.", "Apple": "Apple Inc.",
    "Sony": "Sony Corporation", "LG": "LG Electronics Inc.",
    "Philips": "Koninklijke Philips N.V.", "Bosch": "Robert Bosch GmbH",
    "Nestlé": "Nestlé S.A.", "Unilever": "Unilever PLC",
    "P&G": "Procter & Gamble Company", "Nike": "Nike, Inc.",
    "Adidas": "Adidas AG", "Puma": "Puma SE", "IKEA": "Inter IKEA Group",
    "H&M": "Hennes & Mauritz AB", "Zara": "Inditex S.A.",
    "Panasonic": "Panasonic Holdings Corporation", "Siemens": "Siemens AG",
    "3M": "3M Company", "Colgate": "Colgate-Palmolive Company",
    "Heinz": "The Kraft Heinz Company",
}
RETAILER_BRANDS: list[str] = [
    "Tesco", "Carrefour", "Aldi", "Lidl", "Rewe", "Edeka", "Migros", "Coop",
    "Sainsbury's", "Morrisons", "Mercadona", "Conad", "Esselunga", "Jumbo",
    "Albert Heijn", "Biedronka", "Kaufland", "Penny", "Netto", "Spar",
]
RETAILER_BRAND_FULL: dict[str, str] = {
    "Tesco": "Tesco PLC", "Carrefour": "Carrefour S.A.", "Aldi": "Aldi Einkauf SE",
    "Lidl": "Lidl Stiftung & Co. KG", "Rewe": "REWE Group", "Edeka": "EDEKA Zentrale AG",
    "Migros": "Migros-Genossenschafts-Bund", "Coop": "Coop Genossenschaft",
    "Sainsbury's": "J Sainsbury plc", "Morrisons": "Wm Morrison Supermarkets Ltd",
    "Mercadona": "Mercadona S.A.", "Conad": "Conad Società Cooperativa",
    "Esselunga": "Esselunga S.p.A.", "Jumbo": "Jumbo Groep Holding B.V.",
    "Albert Heijn": "Albert Heijn B.V.", "Biedronka": "Jeronimo Martins Polska S.A.",
    "Kaufland": "Kaufland Stiftung & Co. KG", "Penny": "Penny GmbH & Co. KG",
    "Netto": "Netto Marken-Discount AG", "Spar": "SPAR International",
}

CATEGORIES: list[str] = ["Electronics", "Food & Beverage", "Apparel", "Home & Garden", "Sports Equipment"]
PRODUCT_TEMPLATES: dict[str, list[str]] = {
    "Electronics": ["Smart TV {}", "Laptop Model {}", "Wireless Headphones {}", "Tablet Pro {}", "Smartphone {} Plus"],
    "Food & Beverage": ["Energy Drink Pack {}", "Coffee Blend {}", "Protein Bar {}", "Sparkling Water {}"],
    "Apparel": ["Running Shoes {}", "Sports Jacket {}", "Training Tee {}", "Performance Shorts {}"],
    "Home & Garden": ["Robot Vacuum {}", "Air Purifier {}", "Smart Thermostat {}", "Cordless Drill {}"],
    "Sports Equipment": ["Fitness Tracker {}", "Yoga Mat Pro {}", "Resistance Bands {}", "Dumbbells {}kg"],
}

SPORTS: list[str] = ["Tennis", "Athletics", "Swimming", "Cycling", "Boxing", "Football", "Basketball", "Judo"]
NATIONALITIES: list[str] = list(NAT_MAP.keys())

PLAYER_FIRST: list[str] = [
    "Carlos", "Rafael", "Novak", "Daniil", "Alexander", "Stefanos", "Andrey", "Matteo",
    "Jannik", "Lorenzo", "Holger", "Felix", "Taylor", "Tommy", "Francis", "Denis",
    "Grigor", "Pablo", "Roberto", "Diego", "Lucas", "Marco", "Viktor", "Boris",
    "Sebastian", "Nicolas", "Tomas", "Dominic", "Casper", "David",
]
PLAYER_LAST: list[str] = [
    "Alcaraz", "Nadal", "Djokovic", "Medvedev", "Zverev", "Tsitsipas", "Rublev",
    "Berrettini", "Sinner", "Musetti", "Rune", "Auger-Aliassime", "Fritz", "Paul",
    "Tiafoe", "Shapovalov", "Dimitrov", "Schwartzman", "Klein", "Rossi", "Bublik",
    "Troicki", "Becker", "Lendl", "Wilander", "Edberg", "Berdych", "Thiem",
    "Wawrinka", "Humbert",
]

VENUE_NAMES: list[str] = [
    "National Arena", "Grand Stadium", "Olympic Center", "Metropolitan Arena",
    "City Sports Complex", "Championship Grounds", "Heritage Stadium",
    "Central Arena", "Riverside Sports Park", "Northern Athletic Center",
    "Capital Sports Center", "Harbor Arena", "Premier Grounds",
    "Civic Arena", "Unity Sports Center", "Phoenix Arena", "Apex Stadium",
]
CITIES: list[tuple[str, str]] = [
    ("Madrid", "Spain"), ("Paris", "France"), ("Berlin", "Germany"),
    ("Milan", "Italy"), ("London", "UK"), ("Amsterdam", "Netherlands"),
    ("Warsaw", "Poland"), ("Vienna", "Austria"), ("Lisbon", "Portugal"),
    ("Stockholm", "Sweden"), ("Copenhagen", "Denmark"), ("Brussels", "Belgium"),
    ("Budapest", "Hungary"), ("Prague", "Czech Republic"), ("Oslo", "Norway"),
    ("Helsinki", "Finland"), ("Athens", "Greece"), ("Dublin", "Ireland"),
    ("Zurich", "Switzerland"), ("Barcelona", "Spain"), ("Lyon", "France"),
    ("Hamburg", "Germany"), ("Turin", "Italy"), ("Rotterdam", "Netherlands"),
    ("Krakow", "Poland"), ("Porto", "Portugal"), ("Gothenburg", "Sweden"),
    ("Aarhus", "Denmark"), ("Ghent", "Belgium"), ("Salzburg", "Austria"),
]

ATHLETE_FIRST: list[str] = [
    "Carlos", "Rafael", "Ana", "Maria", "Lucas", "Sofia", "Ivan", "Elena",
    "Marco", "Laura", "Thomas", "Emma", "Julian", "Mia", "Viktor", "Lena",
    "Oliver", "Nina", "Felix", "Sara", "Max", "Clara", "Luis", "Katarina",
    "David", "Petra", "Chris", "Julia",
]
ATHLETE_LAST: list[str] = [
    "Garcia", "Müller", "Rossi", "Dupont", "Smith", "Kowalski", "Santos",
    "Yamamoto", "Chen", "Popescu", "Nkosi", "Eriksson", "Hernandez", "Fischer",
    "Bianchi", "Leclerc", "Taylor", "Novak", "Ferreira", "Andersen", "Russo",
    "Petit", "Martinez", "Weber", "Costa", "Hansen", "Romano", "Morin",
]

PERSON_FIRST: list[str] = [
    "James", "Oliver", "Harry", "Jack", "George", "Noah", "Charlie", "Jacob",
    "Alfie", "Freddie", "Amelia", "Olivia", "Isla", "Emily", "Ava", "Lily",
    "Sophia", "Isabella", "Mia", "Poppy", "William", "Thomas", "Henry",
    "Joshua", "Samuel", "Lewis", "Benjamin", "Daniel", "Grace", "Evie",
    "Alice", "Charlotte", "Hannah", "Jessica", "Sophie",
]
PERSON_LAST: list[str] = [
    "Smith", "Jones", "Williams", "Taylor", "Brown", "Davies", "Evans", "Wilson",
    "Thomas", "Roberts", "Johnson", "Lewis", "Walker", "Robinson", "Wood",
    "Thompson", "White", "Watson", "Jackson", "Wright", "Green", "Harris",
    "Cooper", "King", "Lee", "Martin", "Clarke", "Scott", "Turner", "Hill",
]
DEPARTMENTS: list[str] = ["Engineering", "Marketing", "Finance", "Sales", "HR", "Operations", "Product", "Legal"]
COMPANIES: list[str] = [
    "Acme Corp", "Globex", "Initech", "Umbrella Ltd", "Soylent Inc",
    "Massive Dynamics", "Aperture Science", "Weyland Corp", "Hooli", "Pied Piper",
]

AGE_GROUPS: list[str] = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
SURVEY_COUNTRIES: list[str] = list(NAT_MAP.keys())[:20]

DIRECTOR_FIRST: list[str] = [
    "Martin", "Steven", "Christopher", "Ridley", "James", "David", "Peter",
    "Francis", "Stanley", "Quentin", "Alfonso", "Denis", "Paul", "Luca",
    "Sofia", "Kathryn", "Ava", "Greta", "Claire", "Jane",
]
DIRECTOR_LAST: list[str] = [
    "Scorsetti", "Spielmann", "Nolanberg", "Scottbury", "Camerone", "Finchereux",
    "Jacksonov", "Coppola Jr", "Kubricksen", "Tarantinov", "Cuaronez", "Villeneuven",
    "Andersonsen", "Guadagnino Jr", "Coppola II", "Bigelowska", "DuVernay Jr",
    "Gerwigova", "Denisova", "Campionette",
]
ACTOR_FIRST: list[str] = [
    "Leonardo", "Tom", "Robert", "Brad", "Johnny", "Matt", "Christian", "Ryan",
    "Denzel", "Morgan", "Cate", "Meryl", "Natalie", "Scarlett", "Jennifer",
    "Charlize", "Viola", "Sandra", "Julia", "Anne",
]
ACTOR_LAST: list[str] = [
    "DiCapriox", "Hanksen", "Downeyton", "Pittman", "Deppley", "Damonberg",
    "Baleston", "Goslingov", "Washingtonov", "Freemanix", "Blanchetten",
    "Streepova", "Portmanix", "Johanssonova", "Lawrenceton", "Theronsen",
    "Davisberg", "Bullockova", "Robertsix", "Hathawayova",
]
MOVIE_ADJECTIVES: list[str] = [
    "Dark", "Silent", "Lost", "Broken", "Final", "Hidden", "Last", "Distant",
    "Burning", "Falling", "Rising", "Hollow", "Golden", "Iron", "Crimson",
]
MOVIE_NOUNS: list[str] = [
    "Hour", "Light", "Shadow", "Road", "Storm", "Tide", "Mirror", "Gate",
    "Signal", "Tower", "Bridge", "Frontier", "Echo", "Reign", "Passage",
]
GENRES: list[str] = ["Drama", "Thriller", "Action", "Comedy", "Sci-Fi", "Romance", "Horror", "Documentary"]
LANGUAGES: list[str] = ["EN", "FR", "DE", "IT", "ES", "PT", "JA", "KO", "ZH", "AR"]

SALARY_GRADES: list[dict] = [
    {"grade": "IC1", "level": "Junior", "min_usd": 45000, "max_usd": 65000},
    {"grade": "IC2", "level": "Mid", "min_usd": 65000, "max_usd": 90000},
    {"grade": "IC3", "level": "Senior", "min_usd": 90000, "max_usd": 130000},
    {"grade": "IC4", "level": "Staff", "min_usd": 130000, "max_usd": 175000},
    {"grade": "IC5", "level": "Principal", "min_usd": 175000, "max_usd": 230000},
    {"grade": "M1", "level": "Manager", "min_usd": 100000, "max_usd": 145000},
    {"grade": "M2", "level": "Senior Manager", "min_usd": 145000, "max_usd": 195000},
    {"grade": "M3", "level": "Director", "min_usd": 195000, "max_usd": 260000},
    {"grade": "M4", "level": "VP", "min_usd": 260000, "max_usd": 350000},
    {"grade": "M5", "level": "SVP", "min_usd": 350000, "max_usd": 500000},
]


def abbrev_name(full: str) -> str:
    parts = full.strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return full


def last_first(full: str) -> str:
    parts = full.strip().split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return full


def log_normal_sales(n: int, low: int = 200000, high: int = 80000000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_low = np.log(low)
    log_high = np.log(high)
    mu = (log_low + log_high) / 2
    sigma = (log_high - log_low) / 6
    vals = rng.lognormal(mu, sigma, n)
    return np.clip(vals, low, high).astype(int)


def make_products(n: int = 200) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        cat = random.choice(CATEGORIES)
        template = random.choice(PRODUCT_TEMPLATES[cat])
        suffix = random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) + str(random.randint(10, 99))
        rows.append({
            "product_id": f"PROD{i:03d}",
            "product_name": template.format(suffix),
            "brand": random.choice(BRANDS),
            "category": cat,
        })
    return rows


def make_player_names(n: int) -> list[str]:
    names: list[str] = []
    used: set[str] = set()
    attempts = 0
    while len(names) < n:
        full = f"{random.choice(PLAYER_FIRST)} {random.choice(PLAYER_LAST)}"
        if full not in used:
            used.add(full)
            names.append(full)
        attempts += 1
        if attempts > 5000:
            names.append(f"Player {len(names) + 1:03d}")
    return names[:n]


def make_unique_names(firsts: list[str], lasts: list[str], n: int, prefix: str) -> list[str]:
    names: list[str] = []
    used: set[str] = set()
    i = 1
    while len(names) < n:
        full = f"{random.choice(firsts)} {random.choice(lasts)}"
        if full not in used:
            used.add(full)
            names.append(full)
        else:
            names.append(f"{random.choice(firsts)} {random.choice(lasts)} {prefix}{i}")
            i += 1
        if len(names) >= n:
            break
    return names[:n]


def make_movie_titles(n: int) -> list[str]:
    titles: list[str] = []
    used: set[str] = set()
    while len(titles) < n:
        t = f"The {random.choice(MOVIE_ADJECTIVES)} {random.choice(MOVIE_NOUNS)}"
        if t not in used:
            used.add(t)
            titles.append(t)
        else:
            titles.append(f"{random.choice(MOVIE_ADJECTIVES)} {random.choice(MOVIE_NOUNS)} {len(titles) + 1}")
    return titles[:n]


def gen_sales_tables() -> dict[str, pd.DataFrame]:
    products = make_products(200)
    sales_usd = log_normal_sales(200, seed=1)
    units = np.clip((sales_usd / np.random.uniform(20, 200, 200)).astype(int), 100, 2000000)
    prod_df = pd.DataFrame(products)
    month_factor = 0.9 * np.random.uniform(0.92, 1.08, 200)
    q_factor = 0.88 * np.random.uniform(0.92, 1.08, 200)
    week_factor = 0.9 / 4.33 * np.random.uniform(0.90, 1.10, 200)
    month_factor_eur = 0.9 * np.random.uniform(0.92, 1.08, 200)
    q_factor_eur = 0.88 * np.random.uniform(0.92, 1.08, 200)

    def base(col: str, vals: np.ndarray) -> pd.DataFrame:
        df = prod_df.copy()
        df[col] = vals
        df["units_sold"] = units
        return df

    return {
        "products_sales_usd": base("annual_sales_usd", sales_usd),
        "products_sales_eur": base("annual_sales_eur", (sales_usd * USD_EUR).astype(int)),
        "products_sales_gbp": base("annual_sales_gbp", (sales_usd * USD_GBP).astype(int)),
        "products_sales_jpy": base("annual_sales_jpy", (sales_usd * USD_JPY).astype(int)),
        "products_sales_chf": base("annual_sales_chf", (sales_usd * USD_CHF).astype(int)),
        "products_monthly_usd": prod_df.assign(
            report_month="2023-01",
            monthly_sales_usd=(sales_usd / 12 * month_factor).astype(int),
        ),
        "products_quarterly_usd": prod_df.assign(
            report_quarter="Q1-2023",
            quarterly_sales_usd=(sales_usd / 4 * q_factor).astype(int),
        ),
        "products_weekly_usd": prod_df.assign(
            report_week="2023-W01",
            weekly_sales_usd=(sales_usd * week_factor).astype(int),
        ),
        "products_monthly_eur": prod_df.assign(
            report_month="2023-01",
            monthly_sales_eur=((sales_usd * USD_EUR) / 12 * month_factor_eur).astype(int),
        ),
        "products_quarterly_eur": prod_df.assign(
            report_quarter="Q1-2023",
            quarterly_sales_eur=((sales_usd * USD_EUR) / 4 * q_factor_eur).astype(int),
        ),
    }


def gen_brand_tables() -> dict[str, pd.DataFrame]:
    n = 200
    brand_abbrs = list(BRAND_MAP.keys())
    cats = (CATEGORIES * (n // len(CATEGORIES) + 1))[:n]
    rows = []
    for i in range(1, n + 1):
        ba = random.choice(brand_abbrs)
        rows.append({
            "product_id": f"BP{i:03d}",
            "product_name": f"Product {i:03d}",
            "brand_abbr": ba,
            "brand_full": BRAND_MAP[ba],
            "category": cats[i - 1],
            "annual_sales_usd": random.randint(200000, 50000000),
        })
    df = pd.DataFrame(rows)

    summary_rows = []
    for ba, grp in df.groupby("brand_abbr"):
        summary_rows.append({
            "brand": ba,
            "brand_full": BRAND_MAP[ba],
            "total_products": len(grp),
            "total_sales_usd": int(grp["annual_sales_usd"].sum()),
            "avg_sales_usd": int(grp["annual_sales_usd"].mean()),
        })

    retailer_rows = []
    for i in range(1, 151):
        ba = random.choice(RETAILER_BRANDS)
        retailer_rows.append({
            "store_id": f"ST{i:03d}",
            "store_name": f"Store {i:03d}",
            "brand_abbr": ba,
            "brand_full": RETAILER_BRAND_FULL[ba],
            "country": random.choice(NATIONALITIES),
            "annual_revenue_eur": random.randint(500000, 200000000),
        })
    ret_df = pd.DataFrame(retailer_rows)

    return {
        "products_brand_abbr": df[["product_id", "product_name", "brand_abbr", "category", "annual_sales_usd"]],
        "products_brand_full": df[["product_id", "product_name", "brand_full", "category", "annual_sales_usd"]],
        "brand_category_summary": pd.DataFrame(summary_rows),
        "retailers_brand_abbr": ret_df[["store_id", "store_name", "brand_abbr", "country", "annual_revenue_eur"]],
        "retailers_brand_full": ret_df[["store_id", "store_name", "brand_full", "country", "annual_revenue_eur"]],
    }


def gen_ranking_tables() -> dict[str, pd.DataFrame]:
    n = 120
    player_names = make_player_names(n)
    pts = np.random.randint(120, 2980, n)
    world_ranks = list(range(1, n + 1))
    random.shuffle(world_ranks)
    nats = [random.choice(NATIONALITIES) for _ in range(n)]
    sports = [random.choice(SPORTS) for _ in range(n)]
    ids = [f"PLAY{i:03d}" for i in range(1, n + 1)]

    base = pd.DataFrame({"player_id": ids, "player_name": player_names, "nationality": nats, "sport": sports})

    return {
        "sports_rankings_pts": base.assign(ranking_points=pts, world_rank=world_ranks),
        "sports_rankings_norm": base.assign(
            points_norm=(pts / 29.8).round(1),
            rank_percentile=((n + 1 - np.array(world_ranks)) / n).round(2),
        ),
        "sports_rankings_elo": base.assign(
            elo_rating=(1200 + pts * 0.45 + np.random.normal(0, 30, n)).astype(int),
            rank_elo=world_ranks,
        ),
        "sports_rankings_monthly": base[["player_id", "player_name", "sport"]].assign(
            report_month="2023-06",
            monthly_avg_points=(pts * np.random.uniform(0.88, 1.12, n)).astype(int),
        ),
        "sports_rankings_season": base[["player_id", "player_name", "sport"]].assign(
            season="2022-2023",
            season_avg_points=(pts * np.random.uniform(0.82, 1.18, n)).astype(int),
        ),
        "sports_rankings_career": base[["player_id", "player_name", "sport"]].assign(
            career_span="2015-2023",
            career_total_points=(pts * np.random.uniform(5.5, 9.5, n)).astype(int),
        ),
    }


def gen_venue_tables() -> dict[str, pd.DataFrame]:
    venue_rows = []
    for i in range(1, 151):
        city, country = random.choice(CITIES)
        cap = max(4000, min(95000, int(np.random.lognormal(np.log(30000), 0.6))))
        standing = random.randint(0, min(cap // 6, 15000))
        venue_rows.append({
            "venue_id": f"VEN{i:03d}",
            "venue_name": f"{city} {random.choice(VENUE_NAMES)} {i}",
            "city": city,
            "country": country,
            "capacity_seats": cap,
            "standing_capacity": standing,
        })
    exact_df = pd.DataFrame(venue_rows)

    thousands_df = exact_df[["venue_id", "venue_name", "city", "country"]].copy()
    thousands_df["capacity_thousands"] = (exact_df["capacity_seats"] / 1000).round(1)
    thousands_df["standing_thousands"] = (exact_df["standing_capacity"] / 1000).round(1)

    match_rows = []
    for _, v in exact_df.iterrows():
        for _ in range(random.randint(6, 15)):
            fill = random.uniform(0.55, 1.0)
            match_rows.append({
                "match_id": f"M{len(match_rows) + 1:05d}",
                "venue_id": v["venue_id"],
                "venue_name": v["venue_name"],
                "event_date": f"{random.randint(2021, 2023)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "attendance": int(v["capacity_seats"] * fill),
            })
    match_df = pd.DataFrame(match_rows)

    season_rows = []
    annual_rows = []
    for _, v in exact_df.iterrows():
        att = match_df[match_df["venue_id"] == v["venue_id"]]["attendance"]
        avg = int(att.mean()) if len(att) > 0 else int(v["capacity_seats"] * 0.78)
        cnt = len(att)
        season_rows.append({
            "venue_id": v["venue_id"],
            "venue_name": v["venue_name"],
            "season": "2022-23",
            "avg_season_attendance": avg,
            "match_count": cnt,
        })
        annual_rows.append({
            "venue_id": v["venue_id"],
            "venue_name": v["venue_name"],
            "year": 2022,
            "total_annual_attendance": avg * cnt,
            "match_count": cnt,
        })

    return {
        "venues_capacity_exact": exact_df,
        "venues_capacity_thousands": thousands_df,
        "venues_attendance_match": match_df,
        "venues_attendance_season": pd.DataFrame(season_rows),
        "venues_attendance_annual": pd.DataFrame(annual_rows),
    }


def gen_nationality_tables() -> dict[str, pd.DataFrame]:
    nats = list(NAT_MAP.keys())
    n_ath = 300

    ath_rows = []
    used: set[str] = set()
    for i in range(1, n_ath + 1):
        fn = random.choice(ATHLETE_FIRST)
        ln = random.choice(ATHLETE_LAST)
        full = f"{fn} {ln}"
        if full in used:
            ln = ln + str(i)
            full = f"{fn} {ln}"
        used.add(full)
        nat = random.choice(nats)
        ath_rows.append({
            "athlete_id": f"ATH{i:03d}",
            "athlete_name": full,
            "sport": random.choice(SPORTS),
            "nationality": nat,
            "nationality_iso3": NAT_MAP[nat][0],
            "nationality_iso2": NAT_MAP[nat][1],
        })
    ath_df = pd.DataFrame(ath_rows)

    n_emp = 250
    emp_rows = []
    used2: set[str] = set()
    for i in range(1, n_emp + 1):
        fn = random.choice(PERSON_FIRST)
        ln = random.choice(PERSON_LAST)
        full = f"{fn} {ln}"
        if full in used2:
            full = f"{fn} {ln} {i}"
        used2.add(full)
        nat = random.choice(nats)
        emp_rows.append({
            "employee_id": f"EMP{i:04d}",
            "full_name": full,
            "department": random.choice(DEPARTMENTS),
            "nationality": nat,
            "nationality_iso2": NAT_MAP[nat][1],
        })
    emp_df = pd.DataFrame(emp_rows)

    return {
        "athletes_nat_full": ath_df[["athlete_id", "athlete_name", "sport", "nationality"]],
        "athletes_nat_iso3": ath_df[["athlete_id", "athlete_name", "sport", "nationality_iso3"]],
        "athletes_nat_iso2": ath_df[["athlete_id", "athlete_name", "sport", "nationality_iso2"]],
        "athletes_nat_ioc": ath_df[["athlete_id", "athlete_name", "sport"]].assign(
            nationality_ioc=ath_df["nationality"].map(IOC_MAP)
        ),
        "employees_nationality_full": emp_df[["employee_id", "full_name", "department", "nationality"]],
        "employees_nationality_iso2": emp_df[["employee_id", "full_name", "department", "nationality_iso2"]],
    }


def gen_gender_tables() -> dict[str, pd.DataFrame]:
    n_survey = 500
    survey_rows = []
    for i in range(1, n_survey + 1):
        g = random.choice(["Male"] * 48 + ["Female"] * 48 + ["Non-binary"] * 4)
        survey_rows.append({
            "respondent_id": f"R{i:04d}",
            "age_group": random.choice(AGE_GROUPS),
            "country": random.choice(SURVEY_COUNTRIES),
            "gender": g,
            "gender_abbr": {"Male": "M", "Female": "F", "Non-binary": "NB"}[g],
            "gender_code": {"Male": 1, "Female": 0, "Non-binary": 2}[g],
        })
    s_df = pd.DataFrame(survey_rows)

    n_emp = 250
    emp_rows = []
    for i in range(1, n_emp + 1):
        g = random.choice(["Male"] * 48 + ["Female"] * 48 + ["Non-binary"] * 4)
        emp_rows.append({
            "employee_id": f"EMP{i:04d}",
            "department": random.choice(DEPARTMENTS),
            "gender": g,
            "gender_code": {"Male": 1, "Female": 0, "Non-binary": 2}[g],
        })
    e_df = pd.DataFrame(emp_rows)

    return {
        "survey_gender_full": s_df[["respondent_id", "age_group", "country", "gender"]],
        "survey_gender_abbr": s_df[["respondent_id", "age_group", "country", "gender_abbr"]],
        "survey_gender_binary": s_df[["respondent_id", "age_group", "country", "gender_code"]],
        "employees_gender_full": e_df[["employee_id", "department", "gender"]],
        "employees_gender_code": e_df[["employee_id", "department", "gender_code"]],
    }


def gen_person_tables() -> dict[str, pd.DataFrame]:
    def make_persons(n: int, id_prefix: str) -> pd.DataFrame:
        rows = []
        used: set[str] = set()
        for i in range(1, n + 1):
            fn = random.choice(PERSON_FIRST)
            ln = random.choice(PERSON_LAST)
            full = f"{fn} {ln}"
            if full in used:
                full = f"{fn} {ln} {i}"
                ln = f"{ln} {i}"
            used.add(full)
            rows.append({
                "person_id": f"{id_prefix}{i:04d}",
                "full_name": full,
                "last_first": f"{ln}, {fn}",
                "initials_name": abbrev_name(full),
                "department": random.choice(DEPARTMENTS),
                "company": random.choice(COMPANIES),
            })
        return pd.DataFrame(rows)

    p = make_persons(400, "PER")
    e = make_persons(300, "EMP")

    author_names = make_unique_names(PERSON_FIRST, PERSON_LAST, 200, "AU")
    authors = pd.DataFrame({
        "author_id": [f"AU{i:03d}" for i in range(1, 201)],
        "full_name": author_names,
        "last_first": [last_first(n) for n in author_names],
        "genre": [random.choice(GENRES) for _ in range(200)],
        "country": [random.choice(NATIONALITIES) for _ in range(200)],
    })

    return {
        "persons_fullname": p[["person_id", "full_name", "department", "company"]],
        "persons_lastfirst": p[["person_id", "last_first", "department", "company"]],
        "persons_initials": p[["person_id", "initials_name", "department", "company"]],
        "employees_fullname": e[["person_id", "full_name", "department", "company"]].rename(columns={"person_id": "employee_id"}),
        "employees_lastfirst": e[["person_id", "last_first", "department", "company"]].rename(columns={"person_id": "employee_id"}),
        "authors_fullname": authors[["author_id", "full_name", "genre", "country"]],
        "authors_lastfirst": authors[["author_id", "last_first", "genre", "country"]],
    }


def gen_director_tables() -> dict[str, pd.DataFrame]:
    n_movies = 300
    titles = make_movie_titles(n_movies)
    director_names = make_unique_names(DIRECTOR_FIRST, DIRECTOR_LAST, n_movies, "D")
    actor_pool = make_unique_names(ACTOR_FIRST, ACTOR_LAST, 80, "A")

    movies = pd.DataFrame({
        "movie_id": [f"MV{i:03d}" for i in range(1, n_movies + 1)],
        "name": titles,
        "director": director_names,
        "director_abbr": [abbrev_name(d) for d in director_names],
        "director_lastfirst": [last_first(d) for d in director_names],
        "year": [random.randint(2000, 2023) for _ in range(n_movies)],
        "language": [random.choice(LANGUAGES) for _ in range(n_movies)],
        "genre": [random.choice(GENRES) for _ in range(n_movies)],
    })

    cast_rows = []
    for _, row in movies.iterrows():
        for actor in random.sample(actor_pool, random.randint(2, 5)):
            cast_rows.append({
                "movie_id": row["movie_id"],
                "name": row["name"],
                "year": row["year"],
                "actor": actor,
                "actor_abbr": abbrev_name(actor),
            })
    cast_df = pd.DataFrame(cast_rows)
    cast_summary = cast_df.groupby("actor").agg(total_movies=("name", "count")).reset_index()

    n_series = 200
    series_titles = [f"Series {random.choice(MOVIE_ADJECTIVES)} {i}" for i in range(1, n_series + 1)]
    series_directors = make_unique_names(DIRECTOR_FIRST, DIRECTOR_LAST, n_series, "SD")
    series_df = pd.DataFrame({
        "series_id": [f"SR{i:03d}" for i in range(1, n_series + 1)],
        "name": series_titles,
        "director": series_directors,
        "director_abbr": [abbrev_name(d) for d in series_directors],
        "year": [random.randint(2005, 2023) for _ in range(n_series)],
        "language": [random.choice(LANGUAGES) for _ in range(n_series)],
        "genre": [random.choice(GENRES) for _ in range(n_series)],
    })

    return {
        "movies_directors": movies[["movie_id", "name", "director", "year", "language", "genre"]],
        "movies_director_abbr": movies[["movie_id", "name", "director_abbr", "year", "language", "genre"]],
        "movies_director_lastfirst": movies[["movie_id", "name", "director_lastfirst", "year", "language", "genre"]],
        "movies_cast": cast_df[["movie_id", "name", "year", "actor"]],
        "movies_cast_abbr": cast_df[["movie_id", "name", "year", "actor_abbr"]],
        "cast_summary": cast_summary,
        "series_directors": series_df[["series_id", "name", "director", "year", "language", "genre"]],
        "series_director_abbr": series_df[["series_id", "name", "director_abbr", "year", "language", "genre"]],
    }


def gen_range_tables() -> dict[str, pd.DataFrame]:
    salary_rows = []
    for dept in DEPARTMENTS:
        for grade_info in SALARY_GRADES:
            salary_rows.append({
                "department": dept,
                "grade": grade_info["grade"],
                "level": grade_info["level"],
                "min_salary_usd": grade_info["min_usd"],
                "max_salary_usd": grade_info["max_usd"],
                "midpoint_usd": (grade_info["min_usd"] + grade_info["max_usd"]) // 2,
                "min_salary_eur": int(grade_info["min_usd"] * USD_EUR),
                "max_salary_eur": int(grade_info["max_usd"] * USD_EUR),
                "midpoint_eur": int((grade_info["min_usd"] + grade_info["max_usd"]) // 2 * USD_EUR),
            })
    sal_df = pd.DataFrame(salary_rows)

    age_5yr_bins = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
                    "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"]
    age_10yr_bins = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

    pop_5yr = np.random.randint(200000, 2000000, len(age_5yr_bins))
    pop_10yr = np.array([pop_5yr[i * 2] + pop_5yr[i * 2 + 1] if i * 2 + 1 < len(pop_5yr)
                         else pop_5yr[i * 2] for i in range(len(age_10yr_bins))])

    age_5yr_df = pd.DataFrame({
        "age_range": age_5yr_bins,
        "population_count": pop_5yr,
        "pct_total": (pop_5yr / pop_5yr.sum() * 100).round(2),
    })
    age_10yr_df = pd.DataFrame({
        "age_range": age_10yr_bins,
        "population_count": pop_10yr,
        "pct_total": (pop_10yr / pop_10yr.sum() * 100).round(2),
    })

    price_tiers = ["Budget (0-50)", "Economy (50-100)", "Mid-range (100-250)",
                   "Premium (250-500)", "Luxury (500+)"]
    n_prod = 200
    prod_cats = (CATEGORIES * (n_prod // len(CATEGORIES) + 1))[:n_prod]
    tier_usd = [random.choice(price_tiers) for _ in range(n_prod)]
    tier_eur_map = {
        "Budget (0-50)": "Budget (0-45)",
        "Economy (50-100)": "Economy (45-91)",
        "Mid-range (100-250)": "Mid-range (91-228)",
        "Premium (250-500)": "Premium (228-455)",
        "Luxury (500+)": "Luxury (455+)",
    }

    price_usd_df = pd.DataFrame({
        "product_id": [f"PROD{i:03d}" for i in range(1, n_prod + 1)],
        "category": prod_cats,
        "price_tier": tier_usd,
        "brand": [random.choice(BRANDS) for _ in range(n_prod)],
    })
    price_eur_df = price_usd_df.copy()
    price_eur_df["price_tier"] = [tier_eur_map[t] for t in tier_usd]
    price_eur_df = price_eur_df.rename(columns={"price_tier": "price_tier_eur"})

    return {
        "salary_bands_usd": sal_df[["department", "grade", "level", "min_salary_usd", "max_salary_usd", "midpoint_usd"]],
        "salary_bands_eur": sal_df[["department", "grade", "level", "min_salary_eur", "max_salary_eur", "midpoint_eur"]],
        "age_dist_5yr": age_5yr_df,
        "age_dist_10yr": age_10yr_df,
        "product_price_tiers_usd": price_usd_df,
        "product_price_tiers_eur": price_eur_df,
    }


def gen_employee_salary_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(77)
    grade_weights = [0.25, 0.30, 0.20, 0.12, 0.05, 0.04, 0.02, 0.01, 0.007, 0.003]
    grades = [g["grade"] for g in SALARY_GRADES]
    levels = {g["grade"]: g["level"] for g in SALARY_GRADES}
    mins_usd = {g["grade"]: g["min_usd"] for g in SALARY_GRADES}
    maxs_usd = {g["grade"]: g["max_usd"] for g in SALARY_GRADES}

    n = 250
    names = make_unique_names(PERSON_FIRST, PERSON_LAST, n, "E")
    rows_usd = []
    rows_eur = []
    for i in range(n):
        grade = random.choices(grades, weights=grade_weights)[0]
        lo, hi = mins_usd[grade], maxs_usd[grade]
        annual_usd = int(rng.integers(lo, hi + 1))
        monthly_usd = annual_usd // 12
        rows_usd.append({
            "employee_id": f"ES{i + 1:04d}",
            "full_name": names[i],
            "department": random.choice(DEPARTMENTS),
            "grade": grade,
            "level": levels[grade],
            "annual_salary_usd": annual_usd,
            "monthly_salary_usd": monthly_usd,
        })
        rows_eur.append({
            "employee_id": f"ES{i + 1:04d}",
            "full_name": names[i],
            "department": rows_usd[-1]["department"],
            "grade": grade,
            "level": levels[grade],
            "annual_salary_eur": int(annual_usd * USD_EUR),
            "monthly_salary_eur": int(monthly_usd * USD_EUR),
        })

    return {
        "employee_salary_usd": pd.DataFrame(rows_usd),
        "employee_salary_eur": pd.DataFrame(rows_eur),
    }


def gen_climate_tables() -> dict[str, pd.DataFrame]:
    monthly_rows_c: list[dict] = []
    monthly_rows_f: list[dict] = []
    annual_rows_c: list[dict] = []
    annual_rows_f: list[dict] = []

    for city, country in CITIES:
        lat = CITY_LATITUDE.get(city, 48.0)
        base_c = 25.0 - (lat - 35.0) * 0.75
        swing = 10.0 + (lat - 35.0) * 0.35
        monthly_c = []
        for m in range(12):
            temp = round(base_c - swing * math.cos(2 * math.pi * m / 12) + random.uniform(-0.8, 0.8), 1)
            monthly_c.append(temp)
            temp_f = round(temp * 9 / 5 + 32, 1)
            month_label = f"2023-{m + 1:02d}"
            monthly_rows_c.append({"city": city, "country": country, "month": month_label, "avg_temp_celsius": temp})
            monthly_rows_f.append({"city": city, "country": country, "month": month_label, "avg_temp_fahrenheit": temp_f})
        annual_avg_c = round(sum(monthly_c) / 12, 1)
        annual_avg_f = round(annual_avg_c * 9 / 5 + 32, 1)
        annual_rows_c.append({"city": city, "country": country, "year": 2023, "annual_avg_celsius": annual_avg_c})
        annual_rows_f.append({"city": city, "country": country, "year": 2023, "annual_avg_fahrenheit": annual_avg_f})

    return {
        "city_climate_celsius": pd.DataFrame(monthly_rows_c),
        "city_climate_fahrenheit": pd.DataFrame(monthly_rows_f),
        "city_climate_annual_celsius": pd.DataFrame(annual_rows_c),
        "city_climate_annual_fahrenheit": pd.DataFrame(annual_rows_f),
    }


def save_table(name: str, df: pd.DataFrame) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLES_DIR, f"{name}.csv"), index=False)
    print(f"  {name}.csv ({len(df)} rows)")


# Main guard
if __name__ == "__main__":
    all_tables: dict[str, pd.DataFrame] = {}
    all_tables.update(gen_sales_tables())
    all_tables.update(gen_brand_tables())
    all_tables.update(gen_ranking_tables())
    all_tables.update(gen_venue_tables())
    all_tables.update(gen_nationality_tables())
    all_tables.update(gen_gender_tables())
    all_tables.update(gen_person_tables())
    all_tables.update(gen_director_tables())
    all_tables.update(gen_range_tables())
    all_tables.update(gen_employee_salary_tables())
    all_tables.update(gen_climate_tables())

    for name, df in all_tables.items():
        save_table(name, df)

    print(f"\n{len(all_tables)} tables written to {TABLES_DIR}")

import pandas as pd
import os

DATA_PATH = "data/"
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

print("Veri dosyaları yükleniyor...\n")

# --- POSTINGS (100k örnek) ---
postings = pd.read_csv(
    os.path.join(DATA_PATH, "postings.csv"),
    usecols=[
        "job_id", "company_id", "title", "description",
        "location", "remote_allowed", "formatted_experience_level",
        "formatted_work_type", "normalized_salary", "skills_desc",
        "listed_time"
    ],
    low_memory=False
).sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

# --- DİĞER TABLOLAR ---
job_skills    = pd.read_csv(os.path.join(DATA_PATH, "job_skills.csv"))
skills        = pd.read_csv(os.path.join(DATA_PATH, "skills.csv"))
job_industries= pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
industries    = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))
companies     = pd.read_csv(os.path.join(DATA_PATH, "companies.csv"),
                            usecols=["company_id", "name", "company_size", "country", "city"])
salaries      = pd.read_csv(os.path.join(DATA_PATH, "salaries.csv"))

print(f"postings       : {postings.shape}")
print(f"job_skills     : {job_skills.shape}")
print(f"skills         : {skills.shape}")
print(f"job_industries : {job_industries.shape}")
print(f"industries     : {industries.shape}")
print(f"companies      : {companies.shape}")
print(f"salaries       : {salaries.shape}")

print("\n--- POSTINGS sütunları ---")
print(postings.dtypes)

print("\n--- Eksik değer oranları (postings) ---")
missing = postings.isnull().mean().mul(100).round(1).sort_values(ascending=False)
print(missing[missing > 0])

print("\n--- Deneyim seviyesi dağılımı ---")
print(postings["formatted_experience_level"].value_counts())

print("\n--- Çalışma tipi dağılımı ---")
print(postings["formatted_work_type"].value_counts())

print("\n--- Remote dağılımı ---")
print(postings["remote_allowed"].value_counts(dropna=False))

print("\nYükleme tamamlandı.")
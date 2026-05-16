import pandas as pd
import os

DATA_PATH = "data/"
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

# --- POSTINGS YÜKLEMESİ ---
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

job_skills     = pd.read_csv(os.path.join(DATA_PATH, "job_skills.csv"))
skills         = pd.read_csv(os.path.join(DATA_PATH, "skills.csv"))
job_industries = pd.read_csv(os.path.join(DATA_PATH, "job_industries.csv"))
industries     = pd.read_csv(os.path.join(DATA_PATH, "industries.csv"))
companies      = pd.read_csv(os.path.join(DATA_PATH, "companies.csv"),
                             usecols=["company_id", "name", "company_size", "country", "city"])
salaries       = pd.read_csv(os.path.join(DATA_PATH, "salaries.csv"))

print("Ham veri yüklendi.\n")

# ── 1. POSTINGS TEMİZLEME ─────────────────────────────────────────────────────

# skills_desc neredeyse tamamen boş, kaldır
postings.drop(columns=["skills_desc"], inplace=True)

# remote_allowed: NaN = remote değil (0)
postings["remote_allowed"] = postings["remote_allowed"].fillna(0).astype(int)

# listed_time: unix timestamp → tarih
postings["listed_date"] = pd.to_datetime(postings["listed_time"], unit="ms", errors="coerce")
postings.drop(columns=["listed_time"], inplace=True)

# Lokasyondan eyalet çıkar (ör. "New York, NY" → "NY")
postings["state"] = postings["location"].str.extract(r",\s*([A-Z]{2})\s*$")

# Deneyim seviyesi: boşlukları "Unknown" yap
postings["formatted_experience_level"] = postings["formatted_experience_level"].fillna("Unknown")

# Maaş: aşırı uç değerleri çıkar (yıllık 10k altı veya 1M üstü)
postings.loc[postings["normalized_salary"] < 10_000, "normalized_salary"] = None
postings.loc[postings["normalized_salary"] > 1_000_000, "normalized_salary"] = None

# ── 2. SKİLL TABLOSU: job_id'leri postings ile filtrele ───────────────────────
sample_ids = set(postings["job_id"])
job_skills_f = job_skills[job_skills["job_id"].isin(sample_ids)]
job_skills_full = job_skills_f.merge(skills, on="skill_abr", how="left")

# ── 3. SEKTÖR TABLOSU ─────────────────────────────────────────────────────────
job_industries_f = job_industries[job_industries["job_id"].isin(sample_ids)]
job_industries_full = job_industries_f.merge(industries, on="industry_id", how="left")

# ── 4. ŞİRKET BİLGİSİ postings'e ekle ───────────────────────────────────────
postings = postings.merge(
    companies.rename(columns={"name": "company_name"}),
    on="company_id", how="left"
)

# ── 5. MAAŞ: sadece yıllık ve USD olanlar ─────────────────────────────────────
salaries_clean = salaries[
    (salaries["currency"] == "USD") &
    (salaries["pay_period"] == "YEARLY") &
    (salaries["job_id"].isin(sample_ids))
].copy()
salaries_clean = salaries_clean[salaries_clean["med_salary"].notna() |
                                 salaries_clean["max_salary"].notna()]

# ── ÖZET ──────────────────────────────────────────────────────────────────────
print(f"Temizlenmiş postings  : {postings.shape}")
print(f"Skill kayıtları       : {job_skills_full.shape}")
print(f"Sektör kayıtları      : {job_industries_full.shape}")
print(f"Maaş kayıtları (USD)  : {salaries_clean.shape}")
print(f"\nRemote ilan sayısı    : {postings['remote_allowed'].sum():,}")
print(f"Maaş verisi olan ilan : {postings['normalized_salary'].notna().sum():,}")
print(f"Şirket eşleşen ilan   : {postings['company_name'].notna().sum():,}")
print(f"Eyalet bilgisi olan   : {postings['state'].notna().sum():,}")

print("\nTemizleme tamamlandı.")
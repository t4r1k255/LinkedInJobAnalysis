"""
streamlit_app.py
LinkedIn Job Analysis Dashboard
6 Perspektif: İş Arayan · HR · Eğitim · Yatırımcı · Politika · Araştırmacı

Çalıştırmak için:
    streamlit run streamlit_app.py
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
DATA    = BASE / "data"
OUT     = BASE / "outputs"
MODELS  = BASE / "models"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Job Analysis",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0F1117; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; padding: 4px 0; }
.metric-card {
    background: #1A1D27;
    border: 1px solid #2E3347;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
}
.metric-label { color: #8B92A5; font-size: 13px; margin-bottom: 4px; }
.metric-value { color: #E2E8F0; font-size: 26px; font-weight: 600; }
.metric-delta { font-size: 12px; margin-top: 4px; }
.metric-delta.pos { color: #10B981; }
.metric-delta.neg { color: #F43F5E; }
.section-title {
    color: #C8CDD8;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 4px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2E3347;
}
.insight-box {
    background: #1A1D27;
    border-left: 3px solid #6366F1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    color: #C8CDD8;
    font-size: 14px;
}
.insight-box.green  { border-left-color: #10B981; }
.insight-box.amber  { border-left-color: #F59E0B; }
.insight-box.red    { border-left-color: #F43F5E; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def img(filename, caption=None, use_container_width=True):
    path = OUT / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch" if use_container_width else "content")
    else:
        st.warning(f"Grafik bulunamadı: {filename}")


def metric_card(label, value, delta=None, delta_pos=True):
    delta_html = ""
    if delta:
        cls = "pos" if delta_pos else "neg"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


def insight(text, color="purple"):
    st.markdown(f'<div class="insight-box {color}">{text}</div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


@st.cache_data
def load_postings_summary():
    """Hızlı KPI'lar için özet istatistikler."""
    try:
        post = pd.read_csv(DATA / "postings.csv",
            usecols=["job_id","normalized_salary","currency",
                     "formatted_experience_level","remote_allowed"],
            low_memory=False)
        sal = post[post["normalized_salary"].between(10_000, 1_000_000) &
                   (post["currency"].isin(["USD"]) | post["currency"].isna())]
        return {
            "total_postings": len(post),
            "salary_count":   len(sal),
            "median_salary":  int(sal["normalized_salary"].median()),
            "remote_pct":     round((post["remote_allowed"] == 1.0).mean() * 100, 1),
        }
    except Exception:
        return {"total_postings": 123849, "salary_count": 35604,
                "median_salary": 105000, "remote_pct": 12.3}


@st.cache_resource
def load_pipeline():
    """Salary prediction pipeline — model_02."""
    path = MODELS / "best_salary_pipeline.joblib"
    if not path.exists():
        return None, None
    try:
        obj = joblib.load(path)
        if isinstance(obj, dict):
            return obj.get("pipeline"), obj.get("feature_cols", [])
        return obj, []
    except Exception:
        return None, None


# ── TOP-LEVEL SKILLS LIST (model_02 feature names'den) ───────────────────────
TOP_SKILLS = [
    "Information Technology", "Sales", "Management", "Manufacturing",
    "Engineering", "Health Care Provider", "Business Development",
    "Finance", "Accounting/Auditing", "Administrative",
    "Marketing", "Project Management", "Analyst", "Customer Service",
    "Operations", "Legal", "Research", "Design", "Education", "Consulting",
]

EXP_LEVELS = [
    "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"
]

INDUSTRIES = [
    "Information Technology & Services", "Financial Services",
    "Hospital & Health Care", "Staffing and Recruiting",
    "Internet", "Marketing and Advertising", "Accounting",
    "Software Development", "Management Consulting", "Education Management",
    "Retail", "Construction", "Telecommunications",
]

COMPANY_SIZES = ["1-10", "11-50", "51-200", "201-500", "501-1K", "1K-5K", "5K-10K", "10K+"]
SIZE_MAP_INV  = {"1-10":1,"11-50":2,"51-200":3,"201-500":4,"501-1K":5,"1K-5K":6,"5K-10K":7,"10K+":8}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVİGASYON
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💼 LinkedIn Jobs")
    st.markdown("*123k iş ilanı · Kaggle 2024*")
    st.markdown("---")

    page = st.radio(
        "Perspektif seç:",
        options=[
            "🏠  Ana Sayfa",
            "👤  İş Arayan",
            "🏢  HR / İşe Alım",
            "🎓  Eğitim Kurumu",
            "📈  Yatırımcı",
            "🏛️  Politika Yapıcı",
            "🔬  Araştırmacı",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#4B5563'>Dataset: arshkon/linkedin-job-postings<br>"
        "Models: XGBoost · LightGBM · CatBoost<br>"
        "Best R² = 0.757 (model_03)</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANA SAYFA
# ══════════════════════════════════════════════════════════════════════════════
if "Ana Sayfa" in page:
    st.title("LinkedIn Job Postings — Kapsamlı Analiz")
    st.markdown(
        "**Kaggle** dataset'i kullanılarak 123.849 LinkedIn iş ilanı üzerinde yapılan "
        "çok boyutlu analiz. Maaş tahmini, skill analizi, rekabet metrikleri ve "
        "kariyer perspektifleri."
    )

    summary = load_postings_summary()

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Toplam İlan", f"{summary['total_postings']:,}")
    with c2: metric_card("Maaş Verisi Olan", f"{summary['salary_count']:,}")
    with c3: metric_card("Medyan Maaş (USD)", f"${summary['median_salary']:,}")
    with c4: metric_card("Remote İlan Oranı", f"%{summary['remote_pct']}")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        section("Pazar Genel Görünümü")
        img("01_experience_distribution.png")
    with col2:
        section("Maaş Dağılımı")
        img("05_salary_by_experience.png")

    st.markdown("---")
    section("Proje İçeriği")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **📊 Analizler (13 script)**
        - Market · Skills · LLM (Gemini)
        - Cross-section · Geo · Company
        - Title clustering · Benefits
        - Skill-salary premium
        - Rekabet · Kariyer merdiveni
        - Remote analizi · Salary gap
        """)
    with c2:
        st.markdown("""
        **🤖 Modeller (3 script)**
        - model_01: Baseline R²=0.560
        - model_02: Pipeline + Optuna R²=0.583
        - model_03: Advanced R²=0.757
        - XGBoost · LightGBM · CatBoost
        - SHAP · OOF Ensemble
        """)
    with c3:
        st.markdown("""
        **🔍 6 Perspektif**
        - İş Arayan: maaş tahmini + rehber
        - HR: benchmark + rekabet
        - Eğitim: skill ihtiyaçları
        - Yatırımcı: sektörel büyüme
        - Politika: coğrafi · eşitlik
        - Araştırmacı: model şeffaflığı
        """)


# ══════════════════════════════════════════════════════════════════════════════
# 1. İŞ ARAYAN PERSPEKTİFİ
# ══════════════════════════════════════════════════════════════════════════════
elif "İş Arayan" in page:
    st.title("👤 İş Arayan Perspektifi")
    st.markdown("*Maaş tahmini, kariyer rehberi ve rekabet analizi*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Maaş Tahmini",
        "🎯 Skill Rehberi",
        "📊 Kariyer Merdiveni",
        "⚔️ Rekabet Analizi",
    ])

    # ── TAB 1: Maaş Tahmini ──────────────────────────────────────────────────
    with tab1:
        section("Canlı Maaş Tahmini")
        insight("Model: XGBoost pipeline · R²=0.583 · RMSE=$39,353 · 35k iş ilanı üzerinde eğitildi", "green")

        col_form, col_result = st.columns([1, 1])

        with col_form:
            with st.form("salary_form"):
                exp_level   = st.selectbox("Deneyim Seviyesi", EXP_LEVELS, index=2)
                work_type   = st.selectbox("Çalışma Tipi", ["Full-time", "Contract", "Part-time", "Internship"])
                industry    = st.selectbox("Sektör", INDUSTRIES)
                state       = st.selectbox("Eyalet (ABD)", [
                    "CA", "NY", "TX", "WA", "MA", "IL", "FL", "VA", "GA", "CO",
                    "NC", "NJ", "OH", "PA", "AZ", "MN", "MI", "OR", "MD", "CT",
                ])
                company_size = st.selectbox("Şirket Büyüklüğü", COMPANY_SIZES, index=5)
                is_remote    = st.checkbox("Remote pozisyon")
                skills_sel   = st.multiselect("Skill'ler (birden fazla seçilebilir)", TOP_SKILLS,
                                               default=["Information Technology"])

                submitted = st.form_submit_button("🔮 Maaşı Tahmin Et", use_container_width=True)

        with col_result:
            if submitted:
                pipeline, feature_cols = load_pipeline()
                if pipeline is None:
                    st.error("Model dosyası bulunamadı. `models/best_salary_pipeline.joblib` mevcut olmalı.")
                else:
                    try:
                        row = {c: 0 for c in feature_cols}
                        row["formatted_experience_level"] = exp_level
                        row["formatted_work_type"]        = work_type
                        row["primary_industry"]           = industry
                        row["state_final"]                = state
                        row["company_size"]               = SIZE_MAP_INV.get(company_size, 5)
                        row["remote_flag"]                = int(is_remote)
                        row["is_hourly"]                  = 0
                        row["log_views"]                  = np.log1p(500)
                        row["log_applies"]                = np.log1p(30)
                        row["log_employee_count"]         = np.log1p(1000)
                        row["log_follower_count"]         = np.log1p(5000)

                        for skill in skills_sel:
                            col_name = f"skill_{skill.replace(' ', '_').replace('/', '_')}"
                            if col_name in row:
                                row[col_name] = 1

                        X_pred = pd.DataFrame([row])[feature_cols]
                        pred_log = pipeline.predict(X_pred)[0]
                        pred_salary = np.expm1(pred_log)

                        low  = pred_salary * 0.85
                        high = pred_salary * 1.15

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.success(f"### 💵 Tahmini Yıllık Maaş")
                        st.markdown(f"""
                        <div class="metric-card" style="margin:8px 0">
                            <div class="metric-label">Tahmin</div>
                            <div class="metric-value" style="font-size:36px;color:#10B981">
                                ${pred_salary:,.0f}
                            </div>
                            <div class="metric-delta pos">
                                Aralık: ${low:,.0f} — ${high:,.0f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        | Parametre | Değer |
                        |-----------|-------|
                        | Deneyim | {exp_level} |
                        | Sektör | {industry[:30]} |
                        | Eyalet | {state} |
                        | Şirket büyüklüğü | {company_size} |
                        | Remote | {'Evet' if is_remote else 'Hayır'} |
                        | Seçili skill | {len(skills_sel)} adet |
                        """)
                    except Exception as e:
                        st.error(f"Tahmin hatası: {e}")
            else:
                st.info("👈 Formu doldurup 'Maaşı Tahmin Et' butonuna bas.")
                st.markdown("""
                **Bu form ne yapıyor?**
                - Deneyim, sektör, lokasyon ve skill bilgilerini alır
                - Eğitilmiş XGBoost modeliyle tahmin üretir
                - 35,604 iş ilanı üzerinde eğitilmiştir
                - Tahmin ±%15 güven aralığıyla sunulur
                """)

    # ── TAB 2: Skill Rehberi ─────────────────────────────────────────────────
    with tab2:
        section("Hangi Skill Kaç Dolar Katkı Sağlıyor?")
        insight("Mann-Whitney U testi ile istatistiksel olarak anlamlı (p<0.05) skill primleri", "green")
        col1, col2 = st.columns(2)
        with col1:
            img("42_skill_salary_premium.png", "Skill maaş primi")
        with col2:
            img("43_skill_salary_boxplot.png", "Top 10 skill maaş dağılımı")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            img("44_exp_skill_salary_heatmap.png", "Deneyim × Skill maaş matrisi")
        with col2:
            img("46_skill_count_salary.png", "Skill sayısı arttıkça maaş")

        insight("💡 Ortalama 3-5 skill listeleyen ilanlar en yüksek maaşları sunuyor.", "amber")

    # ── TAB 3: Kariyer Merdiveni ─────────────────────────────────────────────
    with tab3:
        section("Entry'den Executive'e: Kariyer Yolculuğu")

        col1, col2 = st.columns(2)
        with col1:
            img("52_career_ladder_violin.png", "Deneyim seviyesine göre maaş dağılımı")
        with col2:
            img("12_ladder_delta.png", "Seviyeler arası maaş sıçraması")

        img("53_career_ladder_by_industry.png", "Sektör bazlı kariyer merdiveni")

        col1, col2 = st.columns(2)
        with col1:
            img("11_remote_salary_comparison.png", "Remote vs ofis maaş farkı")
        with col2:
            img("12_ladder_role_comparison.png", "Role ailesi × deneyim maaş karşılaştırması")

    # ── TAB 4: Rekabet Analizi ────────────────────────────────────────────────
    with tab4:
        section("Bu Pozisyon Ne Kadar Rekabetçi?")
        insight("Rekabet skoru = başvuru sayısı ÷ görüntülenme sayısı. Yüksek = daha zor.", "red")

        col1, col2 = st.columns(2)
        with col1:
            img("47_most_competitive_titles.png", "En rekabetçi pozisyonlar")
        with col2:
            img("49_competition_by_experience.png", "Deneyim seviyesine göre rekabet")

        col1, col2 = st.columns(2)
        with col1:
            img("51_low_competition_opportunities.png", "Düşük rekabet + yüksek maaş fırsatları")
        with col2:
            img("50_competition_remote_vs_office.png", "Remote vs ofis rekabet")


# ══════════════════════════════════════════════════════════════════════════════
# 2. HR / İŞE ALIM PERSPEKTİFİ
# ══════════════════════════════════════════════════════════════════════════════
elif "HR" in page:
    st.title("🏢 HR / İşe Alım Perspektifi")
    st.markdown("*Pazar benchmark, rekabet analizi ve işe alım stratejisi*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Maaş Benchmark",
        "🏆 Rekabet & Pozisyonlar",
        "🏗️ Şirket Analizi",
        "🎁 Benefit Profili",
    ])

    with tab1:
        section("Pazar Maaş Benchmark")
        col1, col2 = st.columns(2)
        with col1:
            img("05_salary_by_experience.png", "Deneyim seviyesi bazlı medyan maaş")
        with col2:
            img("17_salary_band_experience.png", "Deneyim × Maaş bandı dağılımı")

        img("18_salary_band_industry.png", "Sektör × Maaş bandı dağılımı")

        col1, col2 = st.columns(2)
        with col1:
            img("28_company_size_salary.png", "Şirket büyüklüğü bazlı maaş")
        with col2:
            img("69_salary_gap_by_industry.png", "Sektör bazlı maaş aralığı (müzakere payı)")

        insight("💡 Finans ve yazılım sektörlerinde maaş aralığı genişliği en yüksek — müzakere payı büyük.", "amber")

    with tab2:
        section("Rekabet ve Pozisyon Analizi")
        col1, col2 = st.columns(2)
        with col1:
            img("48_competition_by_industry.png", "Sektör bazlı rekabet skoru")
        with col2:
            img("47_most_competitive_titles.png", "En rekabetçi unvanlar")

        col1, col2 = st.columns(2)
        with col1:
            img("01_experience_distribution.png", "Deneyim seviyesi dağılımı")
        with col2:
            img("20_worktype_experience.png", "Çalışma tipi × Deneyim dağılımı")

        img("45_industry_skill_salary_heatmap.png", "Sektör × Skill maaş matrisi")

    with tab3:
        section("Şirket Profili Analizi")
        col1, col2 = st.columns(2)
        with col1:
            img("27_top_companies.png", "En çok ilan veren şirketler")
        with col2:
            img("30_top_paying_companies.png", "En yüksek maaş ödeyen şirketler")

        col1, col2 = st.columns(2)
        with col1:
            img("29_company_size_industry.png", "Şirket büyüklüğü × Sektör")
        with col2:
            img("31_company_size_experience.png", "Şirket büyüklüğü × Deneyim dağılımı")

    with tab4:
        section("Benefit ve Yan Hak Analizi")
        col1, col2 = st.columns(2)
        with col1:
            img("37_top_benefits.png", "En yaygın 20 benefit")
        with col2:
            img("38_benefits_by_company_size.png", "Şirket büyüklüğüne göre benefit sayısı")

        col1, col2 = st.columns(2)
        with col1:
            img("39_benefits_by_industry.png", "Sektör bazlı benefit dağılımı")
        with col2:
            img("40_benefit_count_salary.png", "Benefit sayısı → Maaş ilişkisi")

        img("41_benefits_by_experience.png", "Deneyim bazlı benefit dağılımı")


# ══════════════════════════════════════════════════════════════════════════════
# 3. EĞİTİM KURUMU PERSPEKTİFİ
# ══════════════════════════════════════════════════════════════════════════════
elif "Eğitim" in page:
    st.title("🎓 Eğitim Kurumu Perspektifi")
    st.markdown("*Müfredat planlaması için piyasa talep analizi*")

    tab1, tab2, tab3 = st.tabs([
        "📚 Skill Talepleri",
        "🎓 Diploma & Nitelik",
        "🚀 Kariyer Başlangıcı",
    ])

    with tab1:
        section("Piyasada En Çok Aranan Skill'ler")
        col1, col2 = st.columns(2)
        with col1:
            img("07_top_20_skills.png", "En çok aranan 20 skill")
        with col2:
            img("09_skill_by_experience_heatmap.png", "Deneyim × Skill dağılımı")

        img("11_skill_by_industry_stacked.png", "Sektör bazlı skill kompozisyonu")

        col1, col2 = st.columns(2)
        with col1:
            img("08_top_industries.png", "En çok ilan veren sektörler")
        with col2:
            img("10_top_paying_skills.png", "En yüksek maaşla ilişkili skill'ler")

        insight("💡 'Information Technology' ve 'Engineering' skill'leri hem en yaygın hem de en yüksek maaşlı.", "green")

    with tab2:
        section("Diploma ve Nitelik Gereksinimleri (LLM Analizi)")
        insight("500 iş ilanı Gemini API ile analiz edilmiştir. Diploma şartı, soft skill ve aciliyet tespiti yapılmıştır.", "amber")

        col1, col2 = st.columns(2)
        with col1:
            img("12_degree_requirement.png", "Diploma şartı dağılımı")
        with col2:
            img("16_degree_by_experience.png", "Deneyim seviyesine göre diploma şartı")

        col1, col2 = st.columns(2)
        with col1:
            img("13_soft_skills.png", "En çok aranan soft skill'ler")
        with col2:
            img("15_tech_vs_nontech_salary.png", "Tech vs Non-Tech maaş farkı")

    with tab3:
        section("Entry-Level Kariyer Başlangıcı")
        col1, col2 = st.columns(2)
        with col1:
            img("12_career_ladder_overall.png", "Kariyer merdiveni genel bakış")
        with col2:
            img("70_salary_gap_by_exp.png", "Deneyim bazlı maaş aralığı")

        col1, col2 = st.columns(2)
        with col1:
            img("55_salary_growth_rate.png", "Sektör bazlı maaş artış hızı")
        with col2:
            img("46_skill_count_salary.png", "Skill çeşitliliği → Maaş etkisi")

        insight("💡 Entry-level'dan Mid-Senior'a geçişte ortalama maaş artışı %45-70 arasında.", "green")


# ══════════════════════════════════════════════════════════════════════════════
# 4. YATIRIMCI PERSPEKTİFİ
# ══════════════════════════════════════════════════════════════════════════════
elif "Yatırımcı" in page:
    st.title("📈 Yatırımcı Perspektifi")
    st.markdown("*Sektörel büyüme, istihdam trendleri ve talent yoğunluğu*")

    tab1, tab2, tab3 = st.tabs([
        "🏭 Sektörel Büyüme",
        "🗺️ Coğrafi Dağılım",
        "💼 Şirket Segmentasyonu",
    ])

    with tab1:
        section("Sektörel İstihdam Hacmi ve Maaş Profili")
        col1, col2 = st.columns(2)
        with col1:
            img("08_top_industries.png", "En çok ilan veren sektörler")
        with col2:
            img("18_salary_band_industry.png", "Sektör × Maaş bandı dağılımı")

        img("29_company_size_industry.png", "Sektör × Şirket büyüklüğü yoğunluk haritası")

        col1, col2 = st.columns(2)
        with col1:
            img("15_tech_vs_nontech_salary.png", "Tech vs Non-Tech maaş karşılaştırması")
        with col2:
            img("48_competition_by_industry.png", "Talent rekabeti sektör bazlı")

        insight("💡 Yazılım ve finansal hizmetler en yüksek maaş sunarken, sağlık sektörü en yüksek ilan hacmine sahip.", "green")

    with tab2:
        section("Coğrafi Talent Yoğunluğu")
        col1, col2 = st.columns(2)
        with col1:
            img("22_state_median_salary.png", "Eyalet bazlı medyan maaş")
        with col2:
            img("04_top_states.png", "En çok ilan veren eyaletler")

        img("25_state_industry_heatmap.png", "Eyalet × Sektör yoğunluk haritası")

        col1, col2 = st.columns(2)
        with col1:
            img("24_top_cities.png", "En çok ilan veren şehirler")
        with col2:
            img("19_state_remote_salary_bubble.png", "Eyalet: Remote oran vs Maaş")

    with tab3:
        section("Şirket Segmentasyonu ve Profili")
        col1, col2 = st.columns(2)
        with col1:
            img("27_top_companies.png", "En aktif işe alım yapan şirketler")
        with col2:
            img("30_top_paying_companies.png", "En yüksek maaş ödeyen şirketler (min 20 ilan)")

        col1, col2 = st.columns(2)
        with col1:
            img("28_company_size_salary.png", "Şirket büyüklüğü → Maaş ilişkisi")
        with col2:
            img("31_company_size_experience.png", "Şirket büyüklüğü × Deneyim profili")

        insight("💡 1K-5K çalışanlı orta ölçekli şirketler hem yüksek maaş hem de yoğun işe alım yapıyor.", "amber")


# ══════════════════════════════════════════════════════════════════════════════
# 5. POLİTİKA YAPICI PERSPEKTİFİ
# ══════════════════════════════════════════════════════════════════════════════
elif "Politika" in page:
    st.title("🏛️ Politika Yapıcı Perspektifi")
    st.markdown("*İstihdam coğrafyası, eşitlik analizi ve çalışma modelleri*")

    tab1, tab2, tab3 = st.tabs([
        "🗺️ Coğrafi Eşitsizlik",
        "⚖️ Maaş Adaleti",
        "🏠 Çalışma Modelleri",
    ])

    with tab1:
        section("Bölgesel İstihdam ve Maaş Eşitsizliği")
        img("22_state_median_salary.png", "Eyalet bazlı medyan maaş farklılıkları")

        col1, col2 = st.columns(2)
        with col1:
            img("26_state_experience.png", "Eyalet bazlı deneyim profili")
        with col2:
            img("23_state_remote_rate.png", "Eyalet bazlı remote oran")

        img("25_state_industry_heatmap.png", "Eyalet × Sektör dağılımı")

        insight("💡 CA ve WA'da medyan maaş ulusal ortalamanın %40+ üzerinde — bölgesel eşitsizlik belirgin.", "red")

    with tab2:
        section("Maaş Adaleti ve Şeffaflık Analizi")
        col1, col2 = st.columns(2)
        with col1:
            img("68_salary_gap_distribution.png", "Maaş aralığı (gap) dağılımı")
        with col2:
            img("71_salary_band_vs_gap.png", "Maaş seviyesi vs gap genişliği")

        img("72_salary_gap_negotiation_map.png", "Sektör × Deneyim müzakere haritası")

        col1, col2 = st.columns(2)
        with col1:
            img("69_salary_gap_by_industry.png", "Sektör bazlı maaş aralığı")
        with col2:
            img("70_salary_gap_by_exp.png", "Deneyim bazlı maaş bandı")

        insight("💡 Medyan maaş aralığı $30,000 (%32) — ilanların yarısında şirket $90k-$122k bandı sunuyor.", "amber")

    with tab3:
        section("Uzaktan Çalışma ve İstihdam Modelleri")
        col1, col2 = st.columns(2)
        with col1:
            img("03_remote_vs_office.png", "Remote vs Ofis/Hibrit dağılımı")
        with col2:
            img("06_remote_by_company_size.png", "Şirket büyüklüğüne göre remote oranı")

        col1, col2 = st.columns(2)
        with col1:
            img("59_remote_by_size_salary.png", "Şirket büyüklüğü: Remote oran ve maaş")
        with col2:
            img("61_remote_exp_salary_grouped.png", "Deneyim × Çalışma tipi maaş karşılaştırması")

        img("60_remote_state_scatter.png", "Eyalet: Remote oran vs Medyan maaş")

        insight("💡 Remote ilanları non-remote ilanlardan ortalama $15-20k daha yüksek maaş sunuyor.", "green")


# ══════════════════════════════════════════════════════════════════════════════
# 6. ARAŞTIRMACI PERSPEKTİFİ
# ══════════════════════════════════════════════════════════════════════════════
elif "Araştırmacı" in page:
    st.title("🔬 Araştırmacı Perspektifi")
    st.markdown("*Model şeffaflığı, SHAP analizi ve metodoloji*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Karşılaştırması",
        "🧠 SHAP Analizi",
        "🔍 Feature Analizi",
        "📖 Metodoloji",
    ])

    with tab1:
        section("3 Model — Performans Karşılaştırması")

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Model 01 R²", "0.560", "Baseline", True)
        with c2: metric_card("Model 02 R²", "0.583", "+4.1% vs baseline", True)
        with c3: metric_card("Model 03 R²", "0.757", "+35.2% vs baseline", True)
        with c4: metric_card("Best RMSE", "$25,225", "Model 03 OOF Ensemble", True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            img("model_01_comparison.png", "Model 01: Baseline karşılaştırma")
        with col2:
            img("model_02_baseline_vs_tuned.png", "Model 02: Baseline vs Tuned")

        col1, col2 = st.columns(2)
        with col1:
            img("model_01_pred_vs_actual.png", "Predicted vs Actual")
        with col2:
            img("model_02_tuning_curves.png", "Optuna tuning eğrileri")

        img("model_01_residuals.png", "Residual analizi")

        insight("💡 Model 03'teki artış: TF-IDF+SVD description features, CV-safe target encoding ve OOF ensemble kombinasyonu.", "green")

    with tab2:
        section("SHAP — Model Açıklanabilirliği")
        insight("LightGBM pipeline · 2000 örneklem · TreeExplainer", "green")

        col1, col2 = st.columns(2)
        with col1:
            img("62_shap_beeswarm.png", "SHAP Beeswarm — en etkili 15 feature")
        with col2:
            img("63_shap_bar.png", "SHAP Bar — Mean |SHAP| importance")

        st.markdown("**SHAP Dependence Plots — Top 4 Feature**")
        col1, col2 = st.columns(2)
        with col1:
            img("65_shap_dependence_follower_count.png", "follower_count etkisi")
            img("67_shap_dependence_desc_svd_6.png", "desc_svd_6 etkisi")
        with col2:
            img("66_shap_dependence_applies.png", "applies etkisi")
            img("68_shap_dependence_exp_title_cluster.png", "exp_title_cluster etkisi")

        insight(
            "⚠️ follower_count ve applies gibi engagement feature'ları yüksek SHAP değeri gösteriyor "
            "ancak bu korelasyonel — nedensel değil. Maaşı açıklayan ilanlar daha fazla görüntüleniyor olabilir.",
            "red"
        )

    with tab3:
        section("Feature Engineering ve Önem Analizi")
        col1, col2 = st.columns(2)
        with col1:
            img("model_01_shap_summary.png", "Model 01: SHAP summary plot")
        with col2:
            img("model_01_shap_importance.png", "Model 01: Feature importance karşılaştırması")

        img("model_02_param_importance.png", "Model 02: Optuna parametre önemi")

        st.markdown("**Model 03 Feature Engineering Katmanları:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Kategorik (OrdinalEncoder)**
            - formatted_experience_level
            - formatted_work_type
            - primary_industry
            - pay_period · company_size
            - state_final · title_cluster
            """)
        with col2:
            st.markdown("""
            **Text (TF-IDF + SVD)**
            - title_text → 25 SVD bileşeni
            - description_clean → 100 SVD bileşeni
            - Leakage removal uygulandı
            """)
        with col3:
            st.markdown("""
            **Target Encoding (CV-safe)**
            - 10 interaction feature
            - city_industry · state_industry
            - exp_industry · exp_title_cluster
            - OOF inverse-RMSE ensemble
            """)

    with tab4:
        section("Metodoloji ve Veri Kaynağı")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📦 Dataset**
            - Kaynak: Kaggle — arshkon/linkedin-job-postings
            - Boyut: 123,849 iş ilanı
            - Tablolar: postings · companies · job_skills ·
              job_industries · industries · benefits ·
              employee_counts · salaries (9 tablo)
            - Tarih: Nisan 2024

            **🔬 Analiz Yaklaşımı**
            - 13 analiz scripti → 72+ grafik
            - Mann-Whitney U testi (skill premium)
            - TF-IDF + K-Means clustering (title)
            - Gemini API (LLM feature extraction)
            - 6 perspektif × hedef kitle odaklı
            """)
        with col2:
            st.markdown("""
            **🤖 Modelleme**
            - Target: normalized_salary (USD, yıllık)
            - 35,604 kullanılabilir satır
            - 5-fold stratified CV
            - Leakage prevention: CV-safe target encoding
            - Ensemble: OOF inverse-RMSE ağırlıklı

            **⚠️ Kısıtlar**
            - Maaş verisi ilanların ~%29'unda mevcut
            - remote_allowed %87 null → interpretation dikkatli
            - Tüm veri ABD merkezli
            - Tek ay snapshot (Nisan 2024)
            - follower_count / applies → korelasyon, nedensellik değil
            """)

        st.markdown("---")
        st.markdown("**📁 Proje Yapısı**")
        st.code("""
LinkedInJobAnalysis/
├── data/               # CSV dosyaları
├── models/             # Eğitilmiş modeller (.joblib)
├── outputs/            # PNG grafikler (72+)
├── analysis_01-13.py   # Analiz scriptleri
├── model_01-03.py      # ML pipeline scriptleri
├── shap_dependence.py  # SHAP görselleştirme
├── utils_progress.py   # Progress bar utility
└── streamlit_app.py    # Bu dashboard
        """, language="text")
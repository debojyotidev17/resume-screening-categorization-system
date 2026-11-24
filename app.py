"""
Resume Screening & Categorization System - Streamlit Web Application
Modern, interactive dashboard for AI-powered resume analysis and categorization.
"""

import os
import sys
import streamlit as st
import pandas as pd

# Add current root directory to sys.path for modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.file_parser import parse_resume_file
from src.text_cleaner import clean_resume, extract_skill_keywords
from src.model_pipeline import ResumeClassifier

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & Categorization",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern CSS Styling
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-weight: 700;
        font-size: 2.25rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.4);
    }
    
    .prediction-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.9;
        margin-bottom: 0.25rem;
    }
    
    .prediction-category {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Category Badges Grid */
    .badge-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    
    .category-badge {
        background-color: #f1f5f9;
        color: #334155;
        font-size: 0.85rem;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 500;
        border: 1px solid #cbd5e1;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """Cache model classifier initialization."""
    return ResumeClassifier()


def main():
    # Header Banner
    st.markdown("""
        <div class="main-header">
            <h1>📄 Resume Screening & Categorization System</h1>
            <p>AI-Powered Multi-Format Resume Parser & Job Category Prediction Pipeline</p>
        </div>
    """, unsafe_allow_html=True)

    classifier = load_classifier()

    # Sidebar Navigation & Setup
    with st.sidebar:
        st.header("⚙️ Application Settings")
        
        # Model Status Badge
        if classifier.is_loaded:
            st.success("🟢 Model Status: Active & Loaded")
        else:
            st.error("🔴 Model Status: Artifacts missing!")
            st.info("Run `python train_model.py` in your terminal to generate trained model files.")

        st.divider()

        # Sample Resume Direct Loader
        st.subheader("💡 Demo Quick-Load")
        st.markdown("Try out the system instantly using pre-loaded sample resumes:")
        
        sample_choice = st.selectbox(
            "Select Sample Resume:",
            ["-- Choose Sample --", "Network Security Engineer", "Health & Fitness"]
        )
        
        st.divider()
        st.markdown("### 📌 About Project")
        st.markdown("""
        - **Algorithm:** Support Vector Machine (SVC)
        - **Vectorization:** TF-IDF
        - **Supported Formats:** PDF, DOCX, TXT
        - **Accuracy:** 99.76%
        - **Author:** [Abhi Suryawanshi](https://github.com/Abhi2006-cloud)
        """)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Resume Screener", "📊 Skill & Text Analytics", "📈 Model Info & Categories"])

    # Handle text source (Upload or Sample)
    resume_text = ""
    file_name_display = ""

    # Check sample selection
    if sample_choice == "Network Security Engineer":
        sample_path = os.path.join("samples", "NetworkSecurityEng_Resume.pdf")
        if not os.path.exists(sample_path):
            sample_path = "NetworkSecurityEng_Resume.pdf"
        if os.path.exists(sample_path):
            resume_text = parse_resume_file(sample_path)
            file_name_display = "Sample: Network Security Engineer Resume"
    elif sample_choice == "Health & Fitness":
        sample_path = os.path.join("samples", "health_fitness_resume.pdf")
        if not os.path.exists(sample_path):
            sample_path = "health_fitness_resume.pdf"
        if os.path.exists(sample_path):
            resume_text = parse_resume_file(sample_path)
            file_name_display = "Sample: Health & Fitness Resume"

    # Tab 1: Resume Screener
    with tab1:
        st.markdown("### Upload Resume for Screening")
        uploaded_file = st.file_uploader(
            "Upload a resume document (PDF, DOCX, or TXT format)",
            type=["pdf", "docx", "txt"]
        )

        if uploaded_file is not None:
            try:
                resume_text = parse_resume_file(uploaded_file, uploaded_file.name)
                file_name_display = uploaded_file.name
            except Exception as e:
                st.error(f"Error parsing file: {str(e)}")

        if resume_text:
            st.success(f"Successfully processed: **{file_name_display}**")
            
            if not classifier.is_loaded:
                st.warning("⚠️ Model classifier is not loaded. Please ensure `clf.pkl`, `tfidf.pkl`, and `encoder.pkl` exist.")
            else:
                # Top Predictions & Confidence Scores
                top_matches = classifier.predict_top_k(resume_text, top_k=3)
                top_cat, top_conf = top_matches[0]

                # Prediction Banner
                st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-title">Predicted Primary Category</div>
                        <div class="prediction-category">{top_cat}</div>
                        <div>Match Confidence Score: <strong>{top_conf:.1f}%</strong></div>
                    </div>
                """, unsafe_allow_html=True)

                # Top 3 Matches Chart
                st.markdown("#### 📊 Top Match Classification Probabilities")
                df_matches = pd.DataFrame(top_matches, columns=["Category", "Confidence (%)"])
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(df_matches.set_index("Category"))
                with col2:
                    st.dataframe(
                        df_matches.style.format({"Confidence (%)": "{:.2f}%"}),
                        use_container_width=True,
                        hide_index=True
                    )

        else:
            st.info("👈 Upload a resume file above or pick a sample from the sidebar to begin screening!")

    # Tab 2: Skill & Text Analytics
    with tab2:
        if resume_text:
            st.markdown("### 🔍 Document & Skill Analytics")
            cleaned_text = clean_resume(resume_text)
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{len(resume_text.split())}</div><div class="metric-label">Total Words</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{len(cleaned_text.split())}</div><div class="metric-label">Cleaned Words</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{len(resume_text)}</div><div class="metric-label">Characters</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{len(extract_skill_keywords(resume_text, 50))}</div><div class="metric-label">Unique Skills</div></div>', unsafe_allow_html=True)

            st.write("")
            st.divider()

            # Key Skill Keywords Chart
            st.markdown("#### 🔑 Top Extracted Skill Terms & Frequencies")
            top_skills = extract_skill_keywords(resume_text, top_n=15)
            if top_skills:
                df_skills = pd.DataFrame(top_skills, columns=["Keyword", "Frequency"])
                col_chart, col_list = st.columns([2, 1])
                with col_chart:
                    st.bar_chart(df_skills.set_index("Keyword"))
                with col_list:
                    st.dataframe(df_skills, use_container_width=True, hide_index=True)

            st.divider()
            
            # Extracted Text Inspector
            with st.expander("👀 Inspect Cleaned vs Raw Resume Text"):
                sub_t1, sub_t2 = st.tabs(["Cleaned Text (Input to Model)", "Raw Extracted Text"])
                with sub_t1:
                    st.text_area("Cleaned Text", cleaned_text, height=250)
                with sub_t2:
                    st.text_area("Raw Text", resume_text, height=250)
        else:
            st.info("Upload or select a sample resume in the first tab to view skill & text analytics.")

    # Tab 3: Model Info & Supported Categories
    with tab3:
        st.markdown("### 🧠 Model Architecture & Supported Job Roles")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.markdown("""
            **🤖 Classifier Model**  
            One-Vs-Rest Support Vector Classifier (SVC) with linear kernel.
            """)
        with info_col2:
            st.markdown("""
            **🔤 Feature Vectorizer**  
            TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
            """)
        with info_col3:
            st.markdown("""
            **🎯 Model Accuracy**  
            **99.76%** empirical accuracy on 20% test dataset split.
            """)

        st.divider()
        st.markdown("#### 🎓 Supported Resume Categories (25+ Job Roles)")
        
        categories = classifier.get_all_categories() if classifier.is_loaded else [
            "Data Science", "Python Developer", "Java Developer", "Web Designing", "HR",
            "Testing", "DevOps Engineer", "Network Security Engineer", "Business Analyst",
            "Sales", "Mechanical Engineer", "Civil Engineer", "Electrical Engineering",
            "Health and fitness", "Operations Manager", "PMO", "Database", "Hadoop",
            "ETL Developer", "DotNet Developer", "Blockchain", "Automation Testing",
            "SAP Developer", "Advocate", "Arts"
        ]

        # Render categories as badges
        badges_html = '<div class="badge-grid">' + "".join([f'<span class="category-badge">✨ {cat}</span>' for cat in sorted(categories)]) + '</div>'
        st.markdown(badges_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

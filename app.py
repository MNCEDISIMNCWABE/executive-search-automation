import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import requests
import logging
import warnings
import re
from typing import List, Optional
import io

# Set up logging and ignore warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API functions
def search_employees_one_row_per_employee_dedup(
    query,
    country_filter=None,
    location_filter=None,
    max_to_fetch=5
):
    """
    Search employees by query, country, and location filters.
    Returns a DataFrame with one row per employee with deduplicated data.
    """
    # 1) Build the Elasticsearch DSL query
    must_clauses = []

    # a) The nested query for experience titles
    must_clauses.append({
        "nested": {
            "path": "member_experience_collection",
            "query": {
                "query_string": {
                    "query": query,
                    "default_field": "member_experience_collection.title",
                    "default_operator": "and"
                }
            }
        }
    })

    # b) If user wants to filter by a specific country (exact match)
    if country_filter:
        must_clauses.append({
            "term": {
                "country": country_filter
            }
        })

    # c) If user wants to filter by a specific location (phrase match)
    if location_filter:
        must_clauses.append({
            "match_phrase": {
                "location": location_filter
            }
        })

    # Combine into a bool query
    payload = {
        "query": {
            "bool": {
                "must": must_clauses
            }
        }
    }

    # 2) Send the search request
    search_url = "https://api.coresignal.com/cdapi/v1/professional_network/employee/search/es_dsl"
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJFZERTQSIsImtpZCI6IjEwYTYwZWRhLWNhNzEtMTIxZS1jY2JhLTBmNjRjMzg4Yjg0ZCJ9.eyJhdWQiOiJheW9iYS5tZSIsImV4cCI6MTc3MzEwNjAyMSwiaWF0IjoxNzQxNTQ5MDY5LCJpc3MiOiJodHRwczovL29wcy5jb3Jlc2lnbmFsLmNvbTo4MzAwL3YxL2lkZW50aXR5L29pZGMiLCJuYW1lc3BhY2UiOiJyb290IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiYXlvYmEubWUiLCJzdWIiOiI5Nzg4ZDg5Ni0yNzBjLTU4NjgtMTY0Mi05MWFiZDk0MGEwODYiLCJ1c2VyaW5mbyI6eyJzY29wZXMiOiJjZGFwaSJ9fQ.BeR_ci_7346iPkfP64QZCwxILa1v1_HGIE1SdhOl9qHtM_HcwiiWIf26DNhcDPl7Bs16JAEfjBntMoyJymtYDA'
    }
    try:
        resp = requests.post(search_url, headers=headers, json=payload)
        resp.raise_for_status()
        employee_ids = resp.json()

        if not isinstance(employee_ids, list):
            logger.error("Unexpected structure in search response")
            return pd.DataFrame()

        # 3) Collect data for each employee ID
        rows = []
        for emp_id in employee_ids[:max_to_fetch]:
            collect_url = f"https://api.coresignal.com/cdapi/v1/professional_network/employee/collect/{emp_id}"
            r = requests.get(collect_url, headers=headers)
            r.raise_for_status()

            employee = r.json()

            # Basic fields
            id_val = employee.get('id')
            name_val = employee.get('name')
            headline_val = employee.get('title')
            location_val = employee.get('location')
            country_val = employee.get('country')
            url_val = employee.get('url')
            industry_val = employee.get('industry')
            experience_count_val = employee.get('experience_count')
            summary_val = employee.get('summary')

            # ----- EXPERIENCE (deduplicate) -----
            raw_exps = employee.get('member_experience_collection', [])
            unique_exps = []
            seen_exps = set()
            for exp in raw_exps:
                key = (
                    exp.get('title', 'N/A'),
                    exp.get('company_name', 'N/A'),
                    exp.get('date_from', 'N/A'),
                    exp.get('date_to', 'N/A')
                )
                if key not in seen_exps:
                    seen_exps.add(key)
                    unique_exps.append(exp)

            experiences_str = "\n".join(
                f"Role: {exp.get('title','N/A')} | Company: {exp.get('company_name','N/A')} "
                f"| From: {exp.get('date_from','N/A')} | To: {exp.get('date_to','N/A')} "
                f"| Duration: {exp.get('duration','N/A')}"
                for exp in unique_exps
            )

            # ----- EDUCATION (deduplicate) -----
            raw_edu = employee.get('member_education_collection', [])
            unique_edu = []
            seen_edu = set()
            for edu in raw_edu:
                key = (
                    edu.get('title', 'N/A'),
                    edu.get('subtitle', 'N/A'),
                    edu.get('date_from', 'N/A'),
                    edu.get('date_to', 'N/A')
                )
                if key not in seen_edu:
                    seen_edu.add(key)
                    unique_edu.append(edu)

            educations_str = "\n".join(
                f"Institution: {edu.get('title','N/A')} | Degree: {edu.get('subtitle','N/A')} "
                f"| From: {edu.get('date_from','N/A')} | To: {edu.get('date_to','N/A')}"
                for edu in unique_edu
            )

            # ----- SKILLS (deduplicate) -----
            raw_skills = employee.get('member_skills_collection', [])
            seen_skills = set()
            for skill_entry in raw_skills:
                skill_name = skill_entry.get('member_skill_list', {}).get('skill', 'N/A')
                if skill_name not in seen_skills:
                    seen_skills.add(skill_name)

            skills_str = ", ".join(seen_skills) if seen_skills else ""

            # Build final row
            row = {
                "ID": id_val,
                "Name": name_val,
                "Headline/Title": headline_val,
                "Location": location_val,
                "Country": country_val,
                "URL": url_val,
                "Industry": industry_val,
                "Experience Count": experience_count_val,
                "Summary": summary_val,
                "Experiences": experiences_str,
                "Educations": educations_str,
                "Skills": skills_str
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        logger.error(f"Error in search_employees: {str(e)}")
        return pd.DataFrame()

# Ranking functions
def build_user_text(row, text_columns: List[str]) -> str:
    """
    Combine relevant text fields into a single string for semantic comparison.
    """
    parts = []
    for col in text_columns:
        val = row.get(col)
        if pd.notnull(val):
            if isinstance(val, list):
                parts.append(' '.join(map(str, val)))
            else:
                parts.append(str(val))
    return " ".join(parts).strip()

def preprocess_text(text: str) -> str:
    """
    Clean and normalize text input by removing emojis, special characters,
    and normalizing whitespace.
    """
    # Remove emojis using Unicode range patterns
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # CJK symbols
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove special characters and punctuation (keep alphanumeric and whitespace)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Convert to lowercase and clean whitespace
    text = text.lower()
    text = ' '.join(text.strip().split())
    return text

def rank_candidates_semantic(
    df_employees: pd.DataFrame,
    job_description: str,
    text_columns: Optional[List[str]] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Rank candidates based on semantic similarity to job description.
    """
    try:
        logger.info("Starting candidate ranking process...")
        
        # Create working copy to avoid modifying original dataframe
        df = df_employees.copy()
        
        if df.empty:
            logger.warning("Empty dataframe provided for ranking")
            return pd.DataFrame()
        
        # Set columns for corpus
        if text_columns is None:
            text_columns = ['Summary', 'Experiences', 'Educations', 
                           'Headline/Title', 'Industry', 'Skills']
            
        # 1) Create combined text for each user
        df['combined_text'] = df.apply(
            lambda x: build_user_text(x, text_columns), 
            axis=1
        )

        # Handle empty texts to avoid encoding issues
        df['combined_text'] = df['combined_text'].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(subset=['combined_text']).reset_index(drop=True)

        if df.empty:
            logger.warning("No valid candidate texts found after preprocessing")
            return pd.DataFrame()

        # 2) Initialize sentence transformer model
        model = SentenceTransformer(model_name)
        
        # 3) Preprocess and embed job description
        clean_jd = preprocess_text(job_description)
        job_embedding = model.encode(clean_jd, convert_to_tensor=True)

        # 4) Embed candidate texts in batches
        user_texts = df['combined_text'].apply(preprocess_text).tolist()
        user_embeddings = model.encode(
            user_texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=True
        )

        # 5) Calculate cosine similarities
        similarities = util.cos_sim(job_embedding, user_embeddings)
        df['similarity_score'] = similarities.cpu().numpy().flatten()
        
        # 6) Sort and return results
        df_sorted = df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
        
        # 7) Format similarity score as percentage
        df_sorted['match_percentage'] = (df_sorted['similarity_score'] * 100).round(1).astype(str) + '%'
        
        return df_sorted

    except Exception as e:
        logger.error(f"Error in ranking candidates: {str(e)}")
        return pd.DataFrame()

# Cache the model to avoid reloading
@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# Function to convert dataframe to Excel for download
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Ranked Candidates', index=False)
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Streamlit UI
def main():
    st.set_page_config(page_title="Candidate Search & Match", layout="wide")
    
    st.title("Candidate Search & Match System")
    st.markdown("Find and rank the best candidates for your job positions")
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'ranked_results' not in st.session_state:
        st.session_state.ranked_results = None
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Search Candidates", "Ranked Results"])
    
    with tab1:
        st.header("Search for Candidates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Search inputs
            search_query = st.text_input("Job Title/Position", 
                                         placeholder="e.g., 'Chief Financial Officer' OR 'CFO'")
            country = st.text_input("Country", placeholder="e.g., South Africa")
            location = st.text_input("City/Location", placeholder="e.g., Johannesburg")
            
            max_results = st.slider("Maximum number of results", 5, 50, 15)
            
            search_button = st.button("Search Candidates")
            
            if search_button and search_query:
                with st.spinner("Searching for candidates..."):
                    # Clear previous results
                    st.session_state.ranked_results = None
                    
                    # Perform search
                    results = search_employees_one_row_per_employee_dedup(
                        query=search_query,
                        country_filter=country if country else None,
                        location_filter=location if location else None,
                        max_to_fetch=max_results
                    )
                    
                    if results.empty:
                        st.error("No candidates found matching your criteria.")
                    else:
                        st.session_state.search_results = results
                        st.success(f"Found {len(results)} candidates!")
        
        with col2:
            # Job description for ranking
            st.subheader("Job Description")
            st.markdown("For best results, provide a detailed job description to rank candidates against:")
            job_description = st.text_area(
                "Enter job description", 
                height=250,
                placeholder="Paste detailed job description here to rank candidates by relevance..."
            )
            
            rank_button = st.button("Rank Candidates")
            
            if rank_button:
                if st.session_state.search_results is None or st.session_state.search_results.empty:
                    st.error("Please search for candidates first before ranking.")
                elif not job_description:
                    st.warning("Please provide a job description for ranking candidates.")
                else:
                    with st.spinner("Ranking candidates..."):
                        # Load model (uses cache)
                        load_model()
                        
                        # Rank candidates
                        ranked_df = rank_candidates_semantic(
                            df_employees=st.session_state.search_results,
                            job_description=job_description,
                            model_name='all-MiniLM-L6-v2'
                        )
                        
                        if ranked_df.empty:
                            st.error("Error occurred during ranking. Please try again.")
                        else:
                            st.session_state.ranked_results = ranked_df
                            st.success("Candidates ranked successfully! View results in the 'Ranked Results' tab.")
        
        # Display search results
        if st.session_state.search_results is not None and not st.session_state.search_results.empty:
            st.subheader("Search Results")
            
            for i, row in st.session_state.search_results.iterrows():
                with st.expander(f"{row['Name']} - {row['Headline/Title']}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**Location:** {row['Location']}")
                        st.markdown(f"**Country:** {row['Country']}")
                        st.markdown(f"**Industry:** {row['Industry']}")
                        st.markdown(f"**Profile URL:** [Link]({row['URL']})")
                    
                    with col2:
                        if pd.notnull(row['Summary']) and row['Summary']:
                            st.markdown("**Summary:**")
                            st.markdown(row['Summary'])
                        
                        if pd.notnull(row['Skills']) and row['Skills']:
                            st.markdown("**Skills:**")
                            st.markdown(row['Skills'])
                    
                    # Display experience details WITHOUT nested expanders
                    if pd.notnull(row['Experiences']) and row['Experiences']:
                        st.markdown("---")
                        st.markdown("### Experience Details")
                        experiences = row['Experiences'].split('\n')
                        for exp in experiences:
                            st.markdown(f"- {exp}")
                    
                    # Display education details WITHOUT nested expanders
                    if pd.notnull(row['Educations']) and row['Educations']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        educations = row['Educations'].split('\n')
                        for edu in educations:
                            st.markdown(f"- {edu}")
    
    with tab2:
        st.header("Ranked Candidates")
        
        if st.session_state.ranked_results is not None and not st.session_state.ranked_results.empty:
            # Add download button for Excel export
            export_columns = [
                'ID', 'Name', 'Headline/Title', 'Location', 'Country', 'URL', 
                'Industry', 'Experience Count', 'Summary', 'Experiences', 
                'Educations', 'Skills', 'combined_text', 'similarity_score'
            ]
            
            # Create a copy with only the requested columns
            export_df = st.session_state.ranked_results[
                [col for col in export_columns if col in st.session_state.ranked_results.columns]
            ].copy()
            
            # Convert similarity score to percentage for better readability in Excel
            if 'similarity_score' in export_df.columns:
                export_df['similarity_score'] = export_df['similarity_score'] * 100
            
            excel_data = to_excel(export_df)
            
            st.download_button(
                label="📥 Download Ranked Candidates (Excel)",
                data=excel_data,
                file_name='ranked_candidates.xlsx',
                mime='application/vnd.ms-excel',
            )
            
            # Show ranking summary
            st.subheader("Match Results")
            
            # Create a bar chart of match percentages
            top_candidates = st.session_state.ranked_results.head(10)
            chart_data = pd.DataFrame({
                'Candidate': top_candidates['Name'],
                'Match Percentage': top_candidates['similarity_score'] * 100
            })
            
            st.bar_chart(chart_data.set_index('Candidate'))
            
            # Show ranked candidates
            for i, row in st.session_state.ranked_results.iterrows():
                with st.expander(f"{row['Name']} - {row['Headline/Title']} (Match: {row['match_percentage']})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**Match Score:** {row['match_percentage']}")
                        st.markdown(f"**Location:** {row['Location']}")
                        st.markdown(f"**Country:** {row['Country']}")
                        st.markdown(f"**Industry:** {row['Industry']}")
                        st.markdown(f"**Profile URL:** [Link]({row['URL']})")
                    
                    with col2:
                        if pd.notnull(row['Summary']) and row['Summary']:
                            st.markdown("**Summary:**")
                            st.markdown(row['Summary'])
                        
                        if pd.notnull(row['Skills']) and row['Skills']:
                            st.markdown("**Skills:**")
                            st.markdown(row['Skills'])
                    
                    # Display experience details WITHOUT nested expanders
                    if pd.notnull(row['Experiences']) and row['Experiences']:
                        st.markdown("---")
                        st.markdown("### Experience Details")
                        experiences = row['Experiences'].split('\n')
                        for exp in experiences:
                            st.markdown(f"- {exp}")
                    
                    # Display education details WITHOUT nested expanders
                    if pd.notnull(row['Educations']) and row['Educations']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        educations = row['Educations'].split('\n')
                        for edu in educations:
                            st.markdown(f"- {edu}")
        else:
            st.info("No ranked results available. Please search for candidates and rank them first.")
            
    # Add footer with instructions
    st.markdown("---")
    st.markdown("""
    **How to use this application:**
    1. Enter a job title/position query in the search box (use OR for multiple terms)
    2. Optionally filter by country and location
    3. Click "Search Candidates" to find matching profiles
    4. Enter a detailed job description to match candidates against
    5. Click "Rank Candidates" to sort by relevance to the job description
    6. View detailed rankings in the "Ranked Results" tab
    7. Download the ranked candidates as an Excel file using the download button
    """)

if __name__ == "__main__":
    main()

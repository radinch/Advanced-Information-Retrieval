import streamlit as st
import sys
import os
import time
import re

# --- Path Setup ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Logic import utils
from Logic.snippet import Snippet
from Logic.preprocess import Preprocessor

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Search Engine",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Beautiful UI ---
st.markdown("""
    <style>
    /* Highlighted search terms */
    .highlight {
        background-color: #ffde59;
        color: #000;
        padding: 0px 4px;
        border-radius: 4px;
        font-weight: 600;
    }
    /* Genre pill badges */
    .genre-pill {
        display: inline-block;
        background-color: #e2e8f0;
        color: #1e293b;
        padding: 4px 10px;
        border-radius: 16px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.85em;
        font-weight: 500;
    }
    .developer-text {
        color: #888888;
        font-size: 0.9em;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Resource Caching ---
@st.cache_resource
def load_nlp_tools():
    preprocessor = Preprocessor()
    snippet_obj = Snippet(
        number_of_words_on_each_side=5,
        normalize_function=preprocessor.normalize,
        remove_stopword_function=preprocessor.remove_stopwords
    )
    return snippet_obj, utils.spell_correction_obj

snippet_obj, spell_correction_tool = load_nlp_tools()

# --- Helper Functions ---
def get_highlighted_texts(description, query):
    """Returns the highlighted snippet and highlighted full description."""
    # Ensure description is a string
    if isinstance(description, list):
        description = " ".join(description)
        
    snippet, _ = snippet_obj.find_snippet(description, query)
    
    highlighted_snippet = snippet
    highlighted_description = description
    
    if "***" in snippet:
        # Convert all ***word*** markers, including cases followed by punctuation.
        highlighted_terms = []

        def replace_marker(match):
            clean_word = match.group(1)
            highlighted_terms.append(clean_word)
            return f"<span class='highlight'>{clean_word}</span>"

        highlighted_snippet = re.sub(r"\*\*\*(.+?)\*\*\*", replace_marker, highlighted_snippet)

        for clean_word in highlighted_terms:
            highlight_html = f"<span class='highlight'>{clean_word}</span>"
            highlighted_description = re.sub(
                rf'(?i)\b{re.escape(clean_word)}\b',
                highlight_html,
                highlighted_description
            )

    return highlighted_snippet, highlighted_description

def display_results(result, search_term, search_time_ms):
    """Handles the rendering of the search result cards."""
    st.success(f"⚡ Search completed in {search_time_ms:.2f} ms")
    
    if not result:
        st.warning("No results found! Try adjusting your search terms or weights.")
        return

    st.markdown(f"### Found {len(result)} results")

    for doc_id, score in result:
        info = utils.get_book_by_id(doc_id, utils.books_dataset)
        
        # Modern Streamlit containers with borders act as neat "cards"
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.subheader(info.get("title", "Unknown Title"))
                if info.get("author"):
                    st.caption(f"**Writer(s):** {info['author']}")
            
            with col2:
                # Using st.metric for a much cleaner display of the score
                st.metric(label="Relevance Score", value=f"{score:.3f}")
            
            # Fetch texts
            description = info.get("description", "")
            snippet, full_description = get_highlighted_texts(description, search_term)
            
            # Show ONLY the snippet outside the expander
            st.markdown(f"... {snippet} ...", unsafe_allow_html=True)
            
            # Expander to show the full description and other details
            with st.expander("View Full Description & Details"):
                st.markdown("**Full Description:**")
                st.markdown(full_description, unsafe_allow_html=True)
                
                st.divider()
                
                # Meta info row
                meta_col1, meta_col2 = st.columns(2)
                with meta_col1:
                    chars = info.get("characters", [])
                    if chars:
                        st.markdown(f"**Characters:** {', '.join(chars)}")
                
                with meta_col2:
                    genres = info.get("genres", [])
                    if genres:
                        # Rendering genres as clean pill badges
                        genres_html = "".join([f"<span class='genre-pill'>{g}</span>" for g in genres])
                        st.markdown(f"**Genres:**<br>{genres_html}", unsafe_allow_html=True)

# --- Main Application ---
def main():
    st.title("📚 Smart Search Engine")
    st.markdown("Search through the Goodreads dataset to find the most relevant entries.")
    st.markdown('<div class="developer-text">Developed By: MIR Team at Sharif University</div>', unsafe_allow_html=True)

    # Search Bar
    search_term = st.text_input("Search Term", placeholder="What are you looking for?...", label_visibility="collapsed")

    # Advanced Search wrapped in an expander, formatted into columns
    with st.expander("⚙️ Advanced Search Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            search_max_num = st.number_input("Maximum number of results", min_value=1, max_value=100, value=10, step=5)
            search_method = st.selectbox("Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25"))
            
        with col2:
            weight_characters = st.slider("Characters Weight", 0.0, 1.0, 1.0, 0.1)
            weight_genres = st.slider("Genres Weight", 0.0, 1.0, 1.0, 0.1)
            weight_description = st.slider("Description Weight", 0.0, 1.0, 1.0, 0.1)

    search_weights = [weight_characters, weight_genres, weight_description]

    # Primary prominent button
    if st.button("Search!", type="primary", use_container_width=True):
        if not search_term.strip():
            st.info("Please enter a search term.")
            return

        corrected_query = spell_correction_tool.spell_check(search_term)

        if corrected_query != search_term:
            st.warning(f"Did you mean: **{corrected_query}**?")
            search_term = corrected_query

        with st.spinner("Searching database..."):
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
            )
            end_time = time.time()
            search_time_ms = (end_time - start_time) * 1000
            
            display_results(result, search_term, search_time_ms)

if __name__ == "__main__":
    main()
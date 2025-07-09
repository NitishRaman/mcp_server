"""
Home page for the MCP Server application.
Minimal landing view; core documentation is available in README.md.
"""
import streamlit as st

# ✅ Page config
st.set_page_config(page_title="🧠 MCP Server", layout="wide")

# ✅ Title
st.title("🧠 Welcome to MCP Server")

# ✅ Intro
st.markdown("""
Welcome to the **MCP Server**, a modular platform to ingest, describe, query, and visualize structured datasets using schema-aware tools and LLM assistance.

📘 For complete documentation, check the sidebar or refer to the `README.md`.
""")

# ✅ Navigation guidance
st.markdown("""
### 🧭 Navigation:
Use the sidebar to switch between modules:
- **Upload & Preview**
- **Description & Schema Generator**
- **Prompt to SQL**
- **Chart Generator**
- **Supabase**
- **Interactive Prompt Chat**
""")

# ✅ Optionally show README.md in expander
with st.expander("📘 View full README.md"):
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("README.md not found.")

# ✅ Optionally show README.md in expander
with st.expander("📘 View full Requirement.txt"):
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("requirements.txt not found.")
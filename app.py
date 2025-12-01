import streamlit as st

from config import init_session_state, setup_page
from data_access import SolarDataExtractor
from ui_tabs import (
    render_kpi_tab,
    render_query_tab,
    render_sidebar,
    render_tables_tab,
    render_upload_tab,
    )
from ui_calculations_v2 import render_calculations_v2_tab
from ui_waterfall_v2 import render_waterfall_tab_v2
from ui_excom_report import render_excom_report_tab

def main():
    init_session_state()
    setup_page()

    extractor = SolarDataExtractor(st.session_state.db_name)

    # Sidebar navigation
    with st.sidebar:
        st.header("📊 Navigation")

        page = st.radio(
            "Select Page",
            ["ExCom Report", "Tables", "KPI Dashboard", "Calculations", "Waterfall"],
            index=0,
            key="page_selector"
        )

        st.divider()
        st.write("### 📁 Data Management")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤\nUpload", use_container_width=True, key="btn_upload"):
                st.session_state.page_selector = "Upload"
                st.rerun()
        with col2:
            if st.button("🔍\nQuery", use_container_width=True, key="btn_query"):
                st.session_state.page_selector = "Query"
                st.rerun()

        st.divider()
        render_sidebar(extractor)

    # Main content area
    if page == "Upload":
        render_upload_tab(st.container(), extractor)
    elif page == "Query":
        render_query_tab(st.container(), extractor)
    elif page == "Tables":
        render_tables_tab(st.container(), extractor)
    elif page == "KPI Dashboard":
        render_kpi_tab(st.container(), extractor)
    elif page == "Calculations":
        render_calculations_v2_tab(st.container(), extractor)
    elif page == "Waterfall":
        render_waterfall_tab_v2(st.container(), extractor)
    elif page == "ExCom Report":
        render_excom_report_tab(st.container(), extractor)

if __name__ == "__main__":
    main()

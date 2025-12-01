import streamlit as st
import pandas as pd

def test_confirm_mapping():
    # Simulate session state
    st.session_state.clear()
    st.session_state['colmap'] = {
        'actual_gen': 'Actual Gen (kWh)',
        'wab': 'Irradiance-based generation',
        'budget_gen': 'Forecast Gen (kWh)',
        'pr_actual': 'Actual PR (%)',
        'pr_budget': 'Forecast PR (%)',
        'availability': 'Availability (%)',
        'capacity': 'kWp',
        'site': 'Site',
        'date': 'Date',
    }
    st.session_state['colmap_confirmed'] = False
    df = pd.DataFrame({
        'Actual Gen (kWh)': [100, 200],
        'Irradiance-based generation': [110, 210],
        'Forecast Gen (kWh)': [120, 220],
        'Actual PR (%)': [90, 91],
        'Forecast PR (%)': [92, 93],
        'Availability (%)': [99, 98],
        'kWp': [10, 10],
        'Site': ['A', 'B'],
        'Date': ['2025-01-01', '2025-01-02'],
    })
    st.session_state['df_ready'] = df
    # Simulate confirm mapping logic
    required = ["actual_gen", "wab", "budget_gen"]
    missing = [k for k in required if not st.session_state['colmap'].get(k)]
    if missing:
        st.write(f"Missing: {missing}")
        return False
    st.session_state['colmap_confirmed'] = True
    st.write("Confirmed!")
    return st.session_state['colmap_confirmed']

if __name__ == "__main__":
    result = test_confirm_mapping()
    print("Confirmed:", result)

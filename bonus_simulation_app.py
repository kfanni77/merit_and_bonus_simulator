import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Load data
df = pd.read_excel("Dummy_HR_Compensation_Dataset.xlsx")

# Simulate team scores and retention risk
np.random.seed(42)
df['TeamScore'] = np.random.randint(1, 6, size=len(df))
df['RetentionRisk'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

# Normalize inputs
df['NormPerformance'] = df['PerformanceRating'] / df['PerformanceRating'].max()
df['NormTeamScore'] = df['TeamScore'] / df['TeamScore'].max()
df['NormRetention'] = df['RetentionRisk']
bonus_mean = df['Bonus'].mean()

# Sidebar: weights
st.sidebar.title("Model Weight Settings")
w_perf = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.5, step=0.05)
w_team = st.sidebar.slider("Team Score Weight", 0.0, 1.0, 0.3, step=0.05)
w_ret = st.sidebar.slider("Retention Risk Weight", 0.0, 1.0, 0.2, step=0.05)

# Normalize weights
total_weight = w_perf + w_team + w_ret
w_perf /= total_weight
w_team /= total_weight
w_ret /= total_weight

# Filters
st.sidebar.title("Filter Data")
selected_dept = st.sidebar.selectbox("Select Department", ["All"] + sorted(df['Department'].unique()))
selected_level = st.sidebar.selectbox("Select Level", ["All"] + sorted(df['Level'].unique()))

filtered_df = df.copy()
if selected_dept != "All":
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
if selected_level != "All":
    filtered_df = filtered_df[filtered_df['Level'] == selected_level]

# Compute modelled bonus
filtered_df['ModelledBonus'] = (
    w_perf * filtered_df['NormPerformance'] +
    w_team * filtered_df['NormTeamScore'] +
    w_ret * filtered_df['NormRetention']
) * bonus_mean * 2

filtered_df['BonusDelta'] = filtered_df['Bonus'] - filtered_df['ModelledBonus']

# KPIs
st.title("Bonus Allocation Simulation Dashboard")

avg_delta = filtered_df['BonusDelta'].mean()
overpaid_pct = (filtered_df['BonusDelta'] > 300).mean() * 100
underpaid_pct = (filtered_df['BonusDelta'] < -300).mean() * 100

st.metric("Average Bonus Delta (EUR)", f"{avg_delta:.0f}")
st.metric("Overpaid (>300 EUR)", f"{overpaid_pct:.1f}%")
st.metric("Underpaid (<-300 EUR)", f"{underpaid_pct:.1f}%")

# Scatter plot
st.subheader("Actual vs. Modelled Bonus")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=filtered_df, x='ModelledBonus', y='Bonus', hue='RetentionRisk', palette='coolwarm', alpha=0.6, ax=ax1)
ax1.plot([filtered_df['ModelledBonus'].min(), filtered_df['ModelledBonus'].max()],
         [filtered_df['ModelledBonus'].min(), filtered_df['ModelledBonus'].max()],
         linestyle='--', color='gray')
ax1.set_title("Actual vs. Modelled Bonus")
ax1.set_xlabel("Modelled Bonus")
ax1.set_ylabel("Actual Bonus")
st.pyplot(fig1)

# Histogram
st.subheader("Distribution of Bonus Deltas")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(filtered_df['BonusDelta'], bins=30, kde=True, ax=ax2, color='steelblue')
ax2.axvline(0, color='black', linestyle='--')
ax2.set_title("Bonus Delta (Actual - Modelled)")
ax2.set_xlabel("Bonus Delta")
st.pyplot(fig2)

# Show table
if st.checkbox("Show Bonus Allocation Table"):
    st.dataframe(filtered_df[['EmployeeID', 'Department', 'Level', 'PerformanceRating', 'TeamScore',
                              'RetentionRisk', 'Bonus', 'ModelledBonus', 'BonusDelta']])

# Download button
def convert_df(df):
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

download_data = convert_df(filtered_df)
st.download_button(
    label="📥 Download Simulation Table as Excel",
    data=download_data,
    file_name="Bonus_Allocation_Simulation_Filtered.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

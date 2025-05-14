import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import statsmodels.api as sm

# --- Simulated Data Load ---
df = pd.read_excel("Dummy_HR_Compensation_Dataset.xlsx")
df['BaseSalary_Original'] = df['BaseSalary']

# Simulate salary bands by level
salary_band_summary = df.groupby('Level')['BaseSalary'].agg(['min', 'median', 'max']).reset_index()
salary_band_summary.columns = ['Level', 'BandMin', 'BandMid', 'BandMax']
df = df.merge(salary_band_summary, on='Level', how='left')

# Calculate band position (0 = min, 0.5 = mid, 1 = max)
df['BandPosition'] = (df['BaseSalary'] - df['BandMin']) / (df['BandMax'] - df['BandMin'])
df['BandPosition'] = df['BandPosition'].clip(0, 1.5)

# User-defined base merit % by performance
st.sidebar.title("Base Merit % by Performance Rating")
base_merit_percent = {
    1: st.sidebar.slider("Rating 1", 0.00, 0.05, 0.00, step=0.005),
    2: st.sidebar.slider("Rating 2", 0.00, 0.05, 0.01, step=0.005),
    3: st.sidebar.slider("Rating 3", 0.00, 0.05, 0.02, step=0.005),
    4: st.sidebar.slider("Rating 4", 0.00, 0.08, 0.03, step=0.005),
    5: st.sidebar.slider("Rating 5", 0.00, 0.12, 0.05, step=0.005),
}
df['BaseMeritPct'] = df['PerformanceRating'].map(base_merit_percent)

# Adjustment factor by salary band position
def adjustment_factor(pos):
    if pos < 0.0:
        return 1.5
    elif pos < 0.5:
        return 1.2
    elif pos <= 1.0:
        return 1.0
    elif pos <= 1.2:
        return 0.5
    else:
        return 0.0

df['BandAdjustment'] = df['BandPosition'].apply(adjustment_factor)
df['AdjustedMeritPct'] = df['BaseMeritPct'] * df['BandAdjustment']
df['RecommendedMeritIncrease'] = df['BaseSalary'] * df['AdjustedMeritPct']

# Sidebar: set total merit budget
MERIT_BUDGET = st.sidebar.number_input("Total Merit Budget (€)", min_value=10000, max_value=1000000, value=200000, step=10000)

# Bonus slider weights
st.sidebar.title("Bonus Allocation Weights")
w_perf = st.sidebar.slider("Performance Weight", 0.0, 1.0, 0.5, step=0.05)
w_team = st.sidebar.slider("Team Score Weight", 0.0, 1.0, 0.3, step=0.05)
w_ret = st.sidebar.slider("Retention Risk Weight", 0.0, 1.0, 0.2, step=0.05)

# Normalize weights
total_weight = w_perf + w_team + w_ret
w_perf /= total_weight
w_team /= total_weight
w_ret /= total_weight

# No scaling — use values based on sliders directly
df['FinalMeritIncrease'] = df['BaseSalary'] * df['AdjustedMeritPct']
df['FinalMeritPct'] = df['AdjustedMeritPct']

# Update BaseSalary with merit increases
df['BaseSalary'] = df['BaseSalary_Original'] + df['FinalMeritIncrease']

# Bonus logic
df['TeamScore'] = np.random.randint(1, 6, size=len(df))
df['RetentionRisk'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
df['NormPerf'] = df['PerformanceRating'] / df['PerformanceRating'].max()
df['NormTeam'] = df['TeamScore'] / df['TeamScore'].max()
df['NormRisk'] = df['RetentionRisk']

# Calculate system bonus recommendation using dynamic slider weights
bonus_mean = df['Bonus'].mean()
df['SystemBonusRecommendation'] = (
    w_perf * df['NormPerf'] + w_team * df['NormTeam'] + w_ret * df['NormRisk']
) * bonus_mean * 2
df['BonusDelta_vs_System'] = df['Bonus'] - df['SystemBonusRecommendation']

# --- Streamlit Layout ---
st.title("Merit & Bonus Simulation App")

col1, col2 = st.columns(2)
col1.metric("Merit Budget", f"€{MERIT_BUDGET:,.0f}")
col2.metric("Simulated Merit Spend", f"€{df['RecommendedMeritIncrease'].sum():,.0f}", delta=f"€{df['RecommendedMeritIncrease'].sum() - MERIT_BUDGET:,.0f}")

st.subheader("Average Scaled Merit % by Performance Rating")
avg_pct_by_rating = df.groupby('PerformanceRating')['AdjustedMeritPct'].mean().round(4).reset_index()
st.dataframe(avg_pct_by_rating)

st.subheader("Bonus Allocation Comparison")
col3, col4 = st.columns(2)
col3.metric("Actual Bonus Total", f"€{df['Bonus'].sum():,.0f}")
col4.metric("System-Recommended Bonus Total", f"€{df['SystemBonusRecommendation'].sum():,.0f}", delta=f"€{df['Bonus'].sum() - df['SystemBonusRecommendation'].sum():,.0f}")

# Visualizations
st.subheader("Bonus Distribution: Actual vs. Recommended")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x='SystemBonusRecommendation', y='Bonus', hue='RetentionRisk', palette='coolwarm', alpha=0.6, ax=ax1)
ax1.plot([df['SystemBonusRecommendation'].min(), df['SystemBonusRecommendation'].max()],
         [df['SystemBonusRecommendation'].min(), df['SystemBonusRecommendation'].max()],
         linestyle='--', color='gray')
ax1.set_title("Actual vs. Recommended Bonus")
ax1.set_xlabel("System-Recommended Bonus (€)")
ax1.set_ylabel("Actual Bonus (€)")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(df['BonusDelta_vs_System'], bins=30, kde=True, color='slateblue', ax=ax2)
ax2.axvline(0, linestyle='--', color='black')
ax2.set_title("Bonus Delta: Actual - System Recommendation")
ax2.set_xlabel("Bonus Delta (€)")
ax2.set_ylabel("Employee Count")
st.pyplot(fig2)

# Gender Pay Gap Calculation
st.subheader("Gender Pay Gap Impact")
avg_salary_gender_before = df.groupby('Gender')['BaseSalary_Original'].mean()
avg_salary_gender_after = df.groupby('Gender')['BaseSalary'].mean()
unadjusted_gpg_before = ((avg_salary_gender_before['Male'] - avg_salary_gender_before['Female']) / avg_salary_gender_before['Male']) * 100
unadjusted_gpg_after = ((avg_salary_gender_after['Male'] - avg_salary_gender_after['Female']) / avg_salary_gender_after['Male']) * 100

st.metric("Unadjusted GPG (Before)", f"{unadjusted_gpg_before:.2f}%")
unadj_gpg_delta = unadjusted_gpg_after - unadjusted_gpg_before
st.metric("Unadjusted GPG (After)", f"{unadjusted_gpg_after:.2f}%",
          delta=f"{unadj_gpg_delta:.2f}%", delta_color="inverse")

# Adjusted Gender Pay Gap via OLS regression
# Adjusted Gender Pay Gap via OLS regression (robust version)
df_encoded = pd.get_dummies(df.copy(), columns=['Gender', 'Level', 'Department'], drop_first=True)

reg_columns = ['TenureYears', 'PerformanceRating'] + [
    col for col in df_encoded.columns if col.startswith('Level_') or col.startswith('Department_') or col.startswith('Gender_')
]

# Coerce everything to numeric and drop bad rows
X = df_encoded[reg_columns].astype(float)
X = sm.add_constant(X)
y_before = pd.to_numeric(df_encoded['BaseSalary_Original'], errors='coerce')
y_after = pd.to_numeric(df_encoded['BaseSalary'], errors='coerce')

# Combine and clean data
regression_data_before = pd.concat([X, y_before], axis=1).dropna()
regression_data_after = pd.concat([X, y_after], axis=1).dropna()

X_before_clean = regression_data_before[X.columns]
y_before_clean = regression_data_before[y_before.name]

X_after_clean = regression_data_after[X.columns]
y_after_clean = regression_data_after[y_after.name]

# Final sanity check
non_numeric_cols = [col for col in X_before_clean.columns if not np.issubdtype(X_before_clean[col].dtype, np.number)]
if non_numeric_cols:
    st.error(f"❌ Non-numeric columns found in X_before_clean: {non_numeric_cols}")
    st.stop()
assert y_before_clean.dtypes == np.float64 or y_before_clean.dtypes == np.int64, "y_before is not numeric"

# Fit models
model_before = sm.OLS(y_before_clean, X_before_clean).fit()
model_after = sm.OLS(y_after_clean, X_after_clean).fit()

adjusted_gap_before = model_before.params.get('Gender_Male', 0)
adjusted_gap_after = model_after.params.get('Gender_Male', 0)

st.metric("Adjusted GPG (Before)", f"€{adjusted_gap_before:.2f}")
gpg_delta = adjusted_gap_after - adjusted_gap_before
delta_color = "inverse"  # Makes increase show as red (bad), decrease as green (good)

st.metric("Adjusted GPG (After)", f"€{adjusted_gap_after:.2f}",
          delta=f"€{gpg_delta:.2f}", delta_color=delta_color)


# Show Data Table
if st.checkbox("Show Detailed Table"):
    st.dataframe(df[['EmployeeID', 'Gender', 'Level', 'PerformanceRating', 'BaseSalary_Original', 'BaseSalary', 'BandPosition', 'ScaledMeritPct', 'ScaledMeritIncrease', 'Bonus', 'SystemBonusRecommendation', 'BonusDelta_vs_System']])

# Download
def convert_df(df):
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

st.download_button(
    label="📥 Download Merit & Bonus Table",
    data=convert_df(df),
    file_name="Merit_Bonus_Simulation.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

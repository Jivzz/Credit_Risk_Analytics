import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main dashboard */
    .stApp {
        background-color: #F4F7FC;
        color: #172B4D;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #17365D !important;
    }

    /* Normal text and captions */
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: #344563;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D5DFED;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(23, 54, 93, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #526581 !important;
    }

    [data-testid="stMetricValue"] {
        color: #17365D !important;
        font-weight: 700;
    }

    [data-testid="stMetricDelta"] {
        color: #344563 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E7EDF6;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #17365D !important;
    }

    /* Dropdowns and multiselect controls */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #AAB9CC !important;
    }

    [data-baseweb="select"] input,
    [data-baseweb="select"] span {
        color: #172B4D !important;
    }

    /* Selected filter tags */
    [data-baseweb="tag"] {
        background-color: #DCE8F8 !important;
    }

    [data-baseweb="tag"] span {
        color: #17365D !important;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #D5DFED;
        border-radius: 8px;
    }

    /* Top-right Streamlit menu and toolbar */
    [data-testid="stToolbar"],
    [data-testid="stMainMenu"],
    [data-testid="stMainMenu"] button,
    header [data-testid="stToolbar"] button {
        color: #FFFFFF !important;
    }

    /* Improve visibility of buttons */
    .stButton button {
        background-color: #17365D;
        color: #FFFFFF !important;
        border-radius: 8px;
        border: none;
    }

    .stButton button:hover {
        background-color: #254F82;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Credit Risk Analytics Dashboard")
st.caption("Portfolio risk overview and unusual account detection")

# Load the prepared datasets
portfolio = pd.read_csv("dashboard_portfolio.csv")
anomalies = pd.read_csv("anomaly_results.csv")
explanations = pd.read_csv("account_risk_explanations.csv")

# Combine risk scores with anomaly results
portfolio = portfolio.drop(
    columns=['anomaly_flag', 'anomaly_score', 'anomaly_status'],
    errors='ignore'
)
data = portfolio.merge(
    anomalies[["account_id", "anomaly_flag", "anomaly_score",
               "anomaly_status"]],
    on="account_id",
    how="left",
    validate="one_to_one"
)

data["risk_pct"] = data["risk_probability"] * 100

# Sidebar filters
st.sidebar.header("Dashboard Filters")

risk_options = sorted(data["risk_segment"].dropna().astype(str).unique())
selected_risks = st.sidebar.multiselect(
    "Risk segment",
    risk_options,
    default=risk_options
)

anomaly_choice = st.sidebar.selectbox(
    "Anomaly status",
    ["All accounts", "Flagged as Unusual", "Normal"]
)

filtered = data[data["risk_segment"].astype(str).isin(selected_risks)]

if anomaly_choice != "All accounts":
    filtered = filtered[filtered["anomaly_status"] == anomaly_choice]

# Portfolio KPIs
c1, c2, c3, c4 = st.columns(4)

c1.metric("Accounts", f"{len(filtered):,}")
c2.metric("Average Predicted Risk", f"{filtered['risk_pct'].mean():.2f}%")
c3.metric("High-Risk Accounts", f"{(filtered['risk_segment'].astype(str) == 'High').sum():,}")
c4.metric("Flagged Anomalies", f"{(filtered['anomaly_flag'] == -1).sum():,}")

st.divider()

# Risk distribution
left, right = st.columns(2)

with left:
    st.subheader("Accounts by Risk Segment")
    risk_counts = (
        filtered["risk_segment"].astype(str)
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("Risk Segment")
        .reset_index(name="Accounts")
    )
    fig = px.bar(
        risk_counts,
        x="Risk Segment",
        y="Accounts",
        color="Risk Segment",
        title="Portfolio Risk Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Anomaly Detection")
    anomaly_counts = (
        filtered["anomaly_status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Accounts")
    )
    fig2 = px.pie(
        anomaly_counts,
        names="Status",
        values="Accounts",
        title="Normal vs Unusual Accounts"
    )
    st.plotly_chart(fig2, use_container_width=True)



# Risk comparisons across portfolio dimensions
st.divider()
st.header("Risk Across Portfolio Dimensions")

dimension_left, dimension_right = st.columns(2)

with dimension_left:
    if "loan_type" in filtered.columns:
        loan_risk = (
            filtered.groupby("loan_type", dropna=False)["risk_pct"]
            .mean()
            .reset_index()
            .sort_values("risk_pct", ascending=False)
        )

        fig_loan_risk = px.bar(
            loan_risk,
            x="loan_type",
            y="risk_pct",
            title="Average Predicted Risk by Loan Type",
            labels={
                "loan_type": "Loan Type",
                "risk_pct": "Average Predicted Risk (%)"
            }
        )
        fig_loan_risk.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_loan_risk, use_container_width=True)

with dimension_right:
    if "employment_type" in filtered.columns:
        employment_risk = (
            filtered.groupby("employment_type", dropna=False)["risk_pct"]
            .mean()
            .reset_index()
            .sort_values("risk_pct", ascending=False)
        )

        fig_employment_risk = px.bar(
            employment_risk,
            x="employment_type",
            y="risk_pct",
            title="Average Predicted Risk by Employment Type",
            labels={
                "employment_type": "Employment Type",
                "risk_pct": "Average Predicted Risk (%)"
            }
        )
        fig_employment_risk.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_employment_risk, use_container_width=True)

# Portfolio exposure analysis
st.divider()
st.header("Portfolio Exposure")

if "loan_amount" in filtered.columns:
    total_exposure = filtered["loan_amount"].sum()
    st.metric("Total Loan Exposure", f"{total_exposure:,.2f}")

    exposure_left, exposure_right = st.columns(2)

    with exposure_left:
        st.subheader("Loan Exposure by Risk Segment")
        exposure_risk = (
            filtered.groupby("risk_segment", observed=False)["loan_amount"]
            .sum()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk Segment")
            .reset_index(name="Loan Exposure")
        )
        fig_exposure = px.bar(
            exposure_risk,
            x="Risk Segment",
            y="Loan Exposure",
            color="Risk Segment",
            title="Total Loan Amount by Risk Segment"
        )
        st.plotly_chart(fig_exposure, use_container_width=True)

    with exposure_right:
        st.subheader("Loan Exposure by Loan Type")
        if "loan_type" in filtered.columns:
            exposure_type = (
                filtered.groupby("loan_type", dropna=False)["loan_amount"]
                .sum()
                .reset_index()
                .sort_values("loan_amount", ascending=False)
            )
            fig_type = px.bar(
                exposure_type,
                x="loan_type",
                y="loan_amount",
                title="Total Loan Amount by Loan Type",
                labels={
                    "loan_type": "Loan Type",
                    "loan_amount": "Loan Exposure"
                }
            )
            fig_type.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_type, use_container_width=True)
else:
    st.info("Loan amount data is unavailable for exposure analysis.")

# Account-level explorer
st.divider()
st.subheader("Account Explorer")

if not filtered.empty:
    account_ids = filtered["account_id"].astype(str).tolist()
    selected_account = st.selectbox("Select an account", account_ids)

    account = filtered[
        filtered["account_id"].astype(str) == selected_account
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Default Risk", f"{account['risk_pct']:.2f}%")
    c2.metric("Risk Segment", str(account["risk_segment"]))
    c3.metric("Anomaly Status", str(account["anomaly_status"]))

    # Model-based account explanation
    account_explanation = explanations[
        explanations["account_id"].astype(str) == selected_account
    ]

    st.subheader("Why is this account receiving this risk score?")

    if not account_explanation.empty:
        explanation = account_explanation.iloc[0]

        risk_col, protective_col = st.columns(2)

        with risk_col:
            st.markdown("#### Key Risk Drivers")
            for i in range(1, 4):
                driver = explanation.get(f"Risk_Driver_{i}")
                if pd.notna(driver):
                    st.markdown(f"- ⚠️ {str(driver).replace('_', ' ')}")

        with protective_col:
            st.markdown("#### Protective Factors")
            for i in range(1, 4):
                factor = explanation.get(f"Protective_Factor_{i}")
                if pd.notna(factor):
                    st.markdown(f"- 🛡️ {str(factor).replace('_', ' ')}")

        st.caption(
            "These are model-based feature contributions, not proof of "
            "causation. Protective factors are associated with a lower "
            "model score relative to the model's reference conditions."
        )
    else:
        st.info("No explanation is available for this account.")

    st.write("**Account details**")
    detail_columns = [
        col for col in [
            "account_id", "loan_type", "loan_amount", "annual_income",
            "employment_type", "bureau_max_days_past_due",
            "late_payments", "repayment_max_days_past_due",
            "emi_shortfall", "anomaly_score"
        ]
        if col in account.index
    ]
    st.dataframe(
        account[detail_columns].to_frame("Value"),
        use_container_width=True
    )

    if account["anomaly_flag"] == -1:
        st.warning(
            "This account has an unusual feature pattern. "
            "Review its details; this flag does not prove fraud or default."
        )
    else:
        st.info("The anomaly detector has not flagged this account.")

else:
    st.info("No accounts match the selected filters.")

st.divider()
st.caption(
    "Prototype: risk segments use illustrative thresholds. "
    "Anomaly flags identify unusual patterns, not confirmed fraud."
)

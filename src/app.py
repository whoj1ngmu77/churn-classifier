import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Classifier", layout="wide", page_icon="⚠")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0008; color: #f0e6e8; }
.stApp { background: linear-gradient(135deg, #0a0008 0%, #120010 50%, #0a0008 100%); }
.stApp > header { display: none; }
header[data-testid="stHeader"] { display: none !important; }
div[data-testid="stDecoration"] { display: none; }
section[data-testid="collapsedControl"] button { color: #e74c3c !important; }
.block-container { padding: 1rem 2.5rem 1.5rem 2.5rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { background: #080006 !important; border-right: 1px solid #2d0015; }
section[data-testid="stSidebar"] .block-container { padding: 0.5rem 1rem !important; }
.metric-card { background: linear-gradient(135deg, #150010 0%, #1a0015 100%); border: 1px solid #2d0015; border-radius: 16px; padding: 1.2rem 1.4rem; position: relative; overflow: hidden; }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #c0392b, #e74c3c, #ff6b6b); border-radius: 16px 16px 0 0; }
.metric-label { font-size: 11px; font-weight: 500; color: #6b3040; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-value { font-size: 24px; font-weight: 700; color: #f0e6e8; line-height: 1; margin-bottom: 4px; }
.metric-sub { font-size: 11px; color: #6b3040; }
.card { background: linear-gradient(135deg, #150010 0%, #1a0015 100%); border: 1px solid #2d0015; border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.card-title { font-size: 13px; font-weight: 600; color: #e74c3c; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
.card-title span { font-size: 11px; color: #6b3040; text-transform: none; letter-spacing: 0; font-weight: 400; }
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div { display: flex; flex-direction: column; gap: 4px; }
div[data-testid="stRadio"] > div > label { display: flex !important; align-items: center; padding: 10px 12px !important; border-radius: 10px !important; font-size: 13px !important; color: #6b3040 !important; cursor: pointer !important; border: none !important; background: transparent !important; }
div[data-testid="stRadio"] > div > label:hover { background: rgba(192,57,43,0.08) !important; color: #f0b8be !important; }
div[data-testid="stRadio"] > div > label[aria-checked="true"] { background: rgba(192,57,43,0.12) !important; color: #e74c3c !important; font-weight: 600 !important; }
div[data-testid="stRadio"] > div > label > div:first-child { display: none !important; }
.stButton button { background: linear-gradient(135deg, #c0392b, #e74c3c) !important; border: none !important; border-radius: 10px !important; color: white !important; font-weight: 700 !important; font-size: 13px !important; }
div[data-testid="stSelectbox"] > div { background: #0d0008 !important; border: 1px solid #2d0015 !important; border-radius: 10px !important; color: #f0e6e8 !important; }
div[data-testid="stNumberInput"] input { background: #0d0008 !important; border: 1px solid #2d0015 !important; border-radius: 10px !important; color: #f0e6e8 !important; }
</style>
""", unsafe_allow_html=True)

CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#6b3040', family='Inter', size=11),
    margin=dict(t=20, b=20, l=0, r=0),
)

COLORS = ['#e74c3c', '#f7b731', '#4ecdc4', '#7c8cff', '#a78bfa', '#ff8a80', '#ffd166', '#06d6a0', '#118ab2', '#fd7b5f']

@st.cache_resource
def load_model():
    with open('../outputs/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../outputs/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/churn_clean.csv')
    fi = pd.read_csv('../data/processed/feature_importance.csv')
    return df, fi

model, scaler = load_model()
df, fi = load_data()

with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem 0;">
      <div style="font-size:18px;font-weight:700;background:linear-gradient(90deg,#e74c3c,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.5px;margin-bottom:2px;">⚠ Churn Classifier</div>
      <div style="font-size:11px;color:#3d1020;">Customer Attrition Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Nav", [
        "▦  Dashboard", "◈  Predict", "△  Model Info", "○  About"
    ], label_visibility="collapsed")

    st.markdown(f"""
    <div style="background:rgba(192,57,43,0.08);border:1px solid rgba(192,57,43,0.2);border-radius:12px;padding:1rem;margin-top:1.5rem;">
      <div style="font-size:11px;font-weight:600;color:#e74c3c;margin-bottom:8px;">MODEL STATS</div>
      <div style="font-size:11px;color:#6b3040;line-height:1.8;">
        Algorithm: XGBoost<br>AUC: 0.8114<br>CV AUC: 0.8228 ± 0.004<br>
        Training samples: {int(len(df)*0.8):,}<br>Features: {df.shape[1]-1}
      </div>
    </div>
    """, unsafe_allow_html=True)

if "Dashboard" in page:
    st.markdown("""
    <div style="font-size:28px;font-weight:700;color:#f0e6e8;letter-spacing:-0.5px;margin-bottom:4px;padding-top:0.5rem;">
      Churn <span style="background:linear-gradient(90deg,#e74c3c,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Risk Intelligence</span>
    </div>
    <div style="font-size:14px;color:#6b3040;margin-bottom:1.5rem;">Telco customer attrition analysis · 7,032 customers</div>
    """, unsafe_allow_html=True)

    total = len(df)
    churned = df['Churn'].sum()
    churn_rate = churned / total * 100
    avg_tenure_churned = df[df['Churn']==1]['tenure'].mean()
    avg_tenure_stayed = df[df['Churn']==0]['tenure'].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Customers</div><div class="metric-value">{total:,}</div><div class="metric-sub">in dataset</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Churn Rate</div><div class="metric-value" style="color:#e74c3c;">{churn_rate:.1f}%</div><div class="metric-sub">{churned:,} customers lost</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Tenure Churned</div><div class="metric-value">{avg_tenure_churned:.0f} mo</div><div class="metric-sub">vs {avg_tenure_stayed:.0f} mo retained</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Model AUC</div><div class="metric-value" style="color:#4ecdc4;">0.8114</div><div class="metric-sub">XGBoost classifier</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card"><div class="card-title">Churn by Contract Type <span>risk coded by color</span></div>', unsafe_allow_html=True)
        contract_churn = df.copy()
        contract_churn['Contract'] = np.where(
            df['Contract_Two year']==True, 'Two year',
            np.where(df['Contract_One year']==True, 'One year', 'Month-to-month')
        )
        cc = contract_churn.groupby('Contract')['Churn'].mean().reset_index()
        cc['Churn'] = cc['Churn'] * 100
        bar_colors = ['#e74c3c' if v > 30 else '#f7b731' if v > 10 else '#4ecdc4' for v in cc['Churn']]
        fig1 = go.Figure(go.Bar(
            x=cc['Contract'], y=cc['Churn'],
            marker_color=bar_colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in cc['Churn']],
            textposition='outside', textfont=dict(color='#6b3040', size=11)
        ))
        fig1.update_layout(**CHART_THEME, height=260, showlegend=False,
                           xaxis=dict(gridcolor='#2d0015', linecolor='#2d0015'),
                           yaxis=dict(gridcolor='#2d0015', linecolor='#2d0015', ticksuffix='%'))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">Top Churn Drivers <span>SHAP importance</span></div>', unsafe_allow_html=True)
        top10 = fi.head(10)
        fig2 = go.Figure(go.Bar(
            x=top10['Importance'], y=top10['Feature'],
            orientation='h',
            marker_color=COLORS[:10], marker_line_width=0
        ))
        fig2.update_layout(**CHART_THEME, height=260, showlegend=False,
                           xaxis=dict(gridcolor='#2d0015', linecolor='#2d0015'),
                           yaxis=dict(gridcolor='#2d0015', linecolor='#2d0015', autorange='reversed'))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Churn Rate by Tenure Group <span>newer customers churn more</span></div>', unsafe_allow_html=True)
    tenure_bins = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0–12 months','13–24 months','25–48 months','49–72 months'])
    tenure_churn = df.groupby(tenure_bins, observed=True)['Churn'].mean().reset_index()
    tenure_churn['Churn'] = tenure_churn['Churn'] * 100
    fig3 = go.Figure(go.Bar(
        x=tenure_churn['tenure'], y=tenure_churn['Churn'],
        marker_color=['#e74c3c', '#f7b731', '#4ecdc4', '#7c8cff'],
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in tenure_churn['Churn']],
        textposition='outside', textfont=dict(color='#6b3040', size=11)
    ))
    fig3.update_layout(**CHART_THEME, height=240, showlegend=False,
                       xaxis=dict(gridcolor='#2d0015', linecolor='#2d0015'),
                       yaxis=dict(gridcolor='#2d0015', linecolor='#2d0015', ticksuffix='%'))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif "Predict" in page:
    st.markdown("""
    <div style="font-size:28px;font-weight:700;color:#f0e6e8;margin-bottom:4px;padding-top:0.5rem;">
      Customer <span style="background:linear-gradient(90deg,#e74c3c,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Risk Predictor</span>
    </div>
    <div style="font-size:14px;color:#6b3040;margin-bottom:1.5rem;">Enter customer details to get churn probability and retention recommendation.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Customer Profile <span>fill in the details</span></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div style="font-size:11px;color:#6b3040;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Account</div>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges (£)", 18, 120, 65)
        total_charges = monthly_charges * tenure
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        st.markdown('<div style="font-size:11px;color:#6b3040;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Plan</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col3:
        st.markdown('<div style="font-size:11px;color:#6b3040;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Add-ons</div>', unsafe_allow_html=True)
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Analyse Churn Risk ↗"):
        binary_map = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}
        services = sum([binary_map.get(x, 0) for x in [phone, multiple_lines, online_security,
                        online_backup, device_protection, tech_support, streaming_tv, streaming_movies, paperless]])

        input_data = {
            'gender': 0, 'SeniorCitizen': 1 if senior=="Yes" else 0,
            'Partner': binary_map[partner], 'Dependents': binary_map[dependents],
            'tenure': tenure, 'PhoneService': binary_map[phone],
            'MultipleLines': binary_map.get(multiple_lines, 0),
            'OnlineSecurity': binary_map.get(online_security, 0),
            'OnlineBackup': binary_map.get(online_backup, 0),
            'DeviceProtection': binary_map.get(device_protection, 0),
            'TechSupport': binary_map.get(tech_support, 0),
            'StreamingTV': binary_map.get(streaming_tv, 0),
            'StreamingMovies': binary_map.get(streaming_movies, 0),
            'PaperlessBilling': binary_map[paperless],
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
            'InternetService_Fiber optic': 1 if internet=="Fiber optic" else 0,
            'InternetService_No': 1 if internet=="No" else 0,
            'Contract_One year': 1 if contract=="One year" else 0,
            'Contract_Two year': 1 if contract=="Two year" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment=="Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if payment=="Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if payment=="Mailed check" else 0,
            'AvgMonthlyCharge': total_charges / (tenure + 1),
            'ChargePerService': monthly_charges / (services + 1),
            'TenureGroup_Loyal': 1 if tenure > 48 else 0,
            'TenureGroup_Mature': 1 if 24 < tenure <= 48 else 0,
            'TenureGroup_New': 1 if tenure <= 12 else 0,
        }

        prob = model.predict_proba(pd.DataFrame([input_data]))[0][1]

        if prob >= 0.7:
            risk_level, risk_color, risk_fill = "High Risk", "#e74c3c", "rgba(231,76,60,0.12)"
            rec = "Immediate intervention required. Call the customer within 24 hours. Offer a contract upgrade with 25-30% discount and a dedicated account manager."
        elif prob >= 0.4:
            risk_level, risk_color, risk_fill = "Medium Risk", "#f7b731", "rgba(247,183,49,0.12)"
            rec = "Proactive outreach recommended within 2 weeks. Send a personalised email with a loyalty reward or service bundle offer."
        else:
            risk_level, risk_color, risk_fill = "Low Risk", "#4ecdc4", "rgba(78,205,196,0.12)"
            rec = "Customer is stable. Standard engagement sufficient. Consider upselling premium services given their loyalty profile."

        col_r, col_g = st.columns([1.2, 0.8])
        with col_r:
            st.markdown(f"""
            <div style="background:{risk_fill};border:1px solid {risk_color}44;border-radius:16px;padding:1.8rem;margin-top:1rem;">
              <div style="display:flex;align-items:center;gap:14px;margin-bottom:1.4rem;padding-bottom:1.2rem;border-bottom:1px solid #2d0015;">
                <div style="width:60px;height:60px;border-radius:14px;background:{risk_fill};border:2px solid {risk_color};display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:700;color:{risk_color};">{int(prob*100)}%</div>
                <div>
                  <div style="font-size:22px;font-weight:700;color:{risk_color};margin-bottom:4px;">{risk_level}</div>
                  <div style="font-size:13px;color:#6b3040;">Churn probability: {prob*100:.1f}%</div>
                </div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.2rem;">
                <div><div style="font-size:10px;color:#6b3040;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Tenure</div><div style="font-size:18px;font-weight:700;color:#f0e6e8;">{tenure} mo</div></div>
                <div><div style="font-size:10px;color:#6b3040;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Monthly</div><div style="font-size:18px;font-weight:700;color:#f0e6e8;">£{monthly_charges}</div></div>
                <div><div style="font-size:10px;color:#6b3040;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Contract</div><div style="font-size:14px;font-weight:700;color:#f0e6e8;">{contract}</div></div>
              </div>
              <div style="background:rgba(0,0,0,0.3);border-radius:12px;padding:14px 16px;border-left:3px solid {risk_color};">
                <div style="font-size:10px;font-weight:600;color:{risk_color};text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Retention Recommendation</div>
                <div style="font-size:13px;color:#d4b0b5;line-height:1.6;">{rec}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_g:
            st.markdown('<div class="card"><div class="card-title">Risk Gauge</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'color': risk_color, 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#6b3040', 'tickfont': {'color': '#6b3040'}},
                    'bar': {'color': risk_color},
                    'bgcolor': '#2d0015',
                    'bordercolor': '#2d0015',
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(78,205,196,0.15)'},
                        {'range': [40, 70], 'color': 'rgba(247,183,49,0.15)'},
                        {'range': [70, 100], 'color': 'rgba(231,76,60,0.15)'}
                    ],
                    'threshold': {'line': {'color': risk_color, 'width': 3}, 'value': prob * 100}
                }
            ))
            fig_gauge.update_layout(**CHART_THEME, height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif "Model Info" in page:
    st.markdown("""
    <div style="font-size:28px;font-weight:700;color:#f0e6e8;margin-bottom:4px;padding-top:0.5rem;">
      Model <span style="background:linear-gradient(90deg,#e74c3c,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Performance</span>
    </div>
    <div style="font-size:14px;color:#6b3040;margin-bottom:1.5rem;">XGBoost classifier evaluation metrics and explainability.</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">ROC-AUC</div><div class="metric-value" style="color:#e74c3c;">0.8114</div><div class="metric-sub">test set</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">CV AUC</div><div class="metric-value" style="color:#4ecdc4;">0.8228</div><div class="metric-sub">5-fold mean</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">F1 Score</div><div class="metric-value" style="color:#f7b731;">0.58</div><div class="metric-sub">churn class</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value" style="color:#7c8cff;">0.78</div><div class="metric-sub">overall</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card"><div class="card-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_data = [[897, 136], [168, 206]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_data, x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
            colorscale=[[0,'#150010'],[0.5,'#922b21'],[1,'#e74c3c']],
            text=cm_data, texttemplate="%{text}",
            textfont=dict(size=22, color='white'), showscale=False
        ))
        fig_cm.update_layout(**CHART_THEME, height=280,
                             xaxis=dict(title='Predicted', linecolor='#2d0015'),
                             yaxis=dict(title='Actual', linecolor='#2d0015'))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">SHAP Feature Importance <span>multi-color ranked</span></div>', unsafe_allow_html=True)
        fig_fi = go.Figure(go.Bar(
            x=fi.head(10)['Importance'], y=fi.head(10)['Feature'],
            orientation='h',
            marker_color=COLORS[:10], marker_line_width=0
        ))
        fig_fi.update_layout(**CHART_THEME, height=280, showlegend=False,
                             xaxis=dict(gridcolor='#2d0015', linecolor='#2d0015'),
                             yaxis=dict(gridcolor='#2d0015', linecolor='#2d0015', autorange='reversed'))
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class="card">
    <div class="card-title">Pipeline Steps</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-top:0.5rem;">
      <div style="text-align:center;padding:1rem;background:#0d0008;border-radius:12px;border:1px solid #2d0015;">
        <div style="font-size:11px;color:#6b3040;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;">Step 1</div>
        <div style="font-size:13px;font-weight:600;color:#e74c3c;">Data Cleaning</div>
        <div style="font-size:11px;color:#6b3040;margin-top:4px;">Fix TotalCharges, drop 11 nulls</div>
      </div>
      <div style="text-align:center;padding:1rem;background:#0d0008;border-radius:12px;border:1px solid #2d0015;">
        <div style="font-size:11px;color:#6b3040;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;">Step 2</div>
        <div style="font-size:13px;font-weight:600;color:#f7b731;">Feature Engineering</div>
        <div style="font-size:11px;color:#6b3040;margin-top:4px;">29 features from 20 raw columns</div>
      </div>
      <div style="text-align:center;padding:1rem;background:#0d0008;border-radius:12px;border:1px solid #2d0015;">
        <div style="font-size:11px;color:#6b3040;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;">Step 3</div>
        <div style="font-size:13px;font-weight:600;color:#4ecdc4;">Model Training</div>
        <div style="font-size:11px;color:#6b3040;margin-top:4px;">XGBoost, RF, LightGBM, LR</div>
      </div>
      <div style="text-align:center;padding:1rem;background:#0d0008;border-radius:12px;border:1px solid #2d0015;">
        <div style="font-size:11px;color:#6b3040;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;">Step 4</div>
        <div style="font-size:13px;font-weight:600;color:#7c8cff;">Evaluation</div>
        <div style="font-size:11px;color:#6b3040;margin-top:4px;">AUC 0.81, 5-fold CV, SHAP</div>
      </div>
    </div>
    </div>""", unsafe_allow_html=True)

elif "About" in page:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#150010 0%,#1a0015 100%);border:1px solid #2d0015;border-radius:20px;padding:2.5rem;margin-bottom:1.5rem;text-align:center;margin-top:0.5rem;">
      <div style="display:inline-block;background:rgba(192,57,43,0.12);color:#e74c3c;padding:4px 14px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:1rem;">⚠ Portfolio Project</div>
      <div style="font-size:28px;font-weight:700;color:#f0e6e8;margin-bottom:0.75rem;">Customer Churn Classifier</div>
      <div style="font-size:14px;color:#6b3040;line-height:1.7;max-width:600px;margin:0 auto;">
        An end-to-end ML pipeline predicting telecom customer churn using XGBoost — with SHAP explainability, 5-fold cross-validation, and a live risk prediction interface.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="card">
          <div class="card-title">Tech Stack</div>
          <div style="margin-top:0.5rem;">
            <span style="display:inline-block;background:rgba(192,57,43,0.08);border:1px solid rgba(192,57,43,0.2);color:#e74c3c;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;margin:4px;">Python</span>
            <span style="display:inline-block;background:rgba(192,57,43,0.08);border:1px solid rgba(192,57,43,0.2);color:#e74c3c;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;margin:4px;">Scikit-learn</span>
            <span style="display:inline-block;background:rgba(247,183,49,0.08);border:1px solid rgba(247,183,49,0.2);color:#f7b731;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;margin:4px;">XGBoost</span>
            <span style="display:inline-block;background:rgba(247,183,49,0.08);border:1px solid rgba(247,183,49,0.2);color:#f7b731;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;margin:4px;">LightGBM</span>
            <span style="display:inline-block;background:rgba(78,205,196,0.08);border:1px solid rgba(78,205,196,0.2);color:#4ecdc4;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;margin:4px;">SHAP</span>
            <span style="display:inline-block;background:rgba(78,205,196,0.08);border:1px solid rgba(78,205,196,0.2);color:#4ecdc4;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;margin:4px;">Plotly</span>
            <span style="display:inline-block;background:rgba(124,140,255,0.08);border:1px solid rgba(124,140,255,0.2);color:#7c8cff;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;margin:4px;">Streamlit</span>
            <span style="display:inline-block;background:rgba(124,140,255,0.08);border:1px solid rgba(124,140,255,0.2);color:#7c8cff;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;margin:4px;">Pandas</span>
          </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="card">
          <div class="card-title">Key Findings</div>
          <div style="display:flex;flex-direction:column;gap:10px;margin-top:0.5rem;">
            <div style="font-size:13px;color:#d4b0b5;padding:8px 12px;background:#0d0008;border-radius:8px;border-left:3px solid #e74c3c;">Month-to-month customers churn at 42% vs 3% on two-year contracts</div>
            <div style="font-size:13px;color:#d4b0b5;padding:8px 12px;background:#0d0008;border-radius:8px;border-left:3px solid #f7b731;">Tenure is #1 predictor — new customers are highest risk</div>
            <div style="font-size:13px;color:#d4b0b5;padding:8px 12px;background:#0d0008;border-radius:8px;border-left:3px solid #4ecdc4;">Fiber optic customers churn more despite paying more</div>
            <div style="font-size:13px;color:#d4b0b5;padding:8px 12px;background:#0d0008;border-radius:8px;border-left:3px solid #7c8cff;">Engineered ChargePerService ranked #3 in SHAP importance</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="text-align:center;padding:2rem;">
      <div style="font-size:13px;color:#6b3040;margin-bottom:0.5rem;">Built by</div>
      <div style="font-size:20px;font-weight:700;background:linear-gradient(90deg,#e74c3c,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Gayathri Menon</div>
    </div>
    """, unsafe_allow_html=True)

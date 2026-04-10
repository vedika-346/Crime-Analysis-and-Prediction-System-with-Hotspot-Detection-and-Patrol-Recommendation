import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CrimeLens | Chicago Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
D = lambda f: os.path.join(BASE_DIR, "data", f)
M = lambda f: os.path.join(BASE_DIR, f)

# ── Design System ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:        #06080f;
    --surface:   #0c1120;
    --surface2:  #111827;
    --border:    rgba(255,255,255,0.07);
    --border2:   rgba(255,255,255,0.12);
    --accent:    #4f7cff;
    --accent2:   #00e5cc;
    --danger:    #ff4d6d;
    --warning:   #ffb347;
    --success:   #3dffa0;
    --text:      #e8edf5;
    --muted:     #5a6a85;
    --mono:      'DM Mono', monospace;
    --display:   'Syne', sans-serif;
    --body:      'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--body);
    background: var(--bg) !important;
    color: var(--text);
}
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }
.main .block-container > div:first-child { min-height: 100vh; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 2px;
    display: flex;
    flex-direction: column;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child { display: none !important; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    width: 100%; padding: 0; margin: 0;
    background: transparent !important; border: none !important; outline: none !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: var(--body); font-size: 13px; font-weight: 400; color: var(--muted);
    padding: 8px 12px; border-radius: 6px; cursor: pointer; transition: all 0.15s;
    display: flex !important; align-items: center !important;
    width: 100%; box-sizing: border-box; letter-spacing: 0.01em; margin: 0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover { background: var(--surface2); color: var(--text); }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:has(input:checked) label {
    background: rgba(79,124,255,0.12) !important; color: var(--accent) !important; font-weight: 500;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:has(input:checked) label::before {
    content: ''; display: inline-block; width: 3px; height: 14px;
    background: var(--accent); border-radius: 2px; margin-right: 10px; flex-shrink: 0;
}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:not(:has(input:checked)) label::before {
    content: ''; display: inline-block; width: 3px; height: 14px;
    background: transparent; border-radius: 2px; margin-right: 10px; flex-shrink: 0;
}
[data-testid="stSidebar"] .stRadio > div > label { display: none; }

[data-testid="metric-container"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 20px; transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: var(--border2); }
[data-testid="metric-container"] label {
    color: var(--muted) !important; font-family: var(--mono) !important;
    font-size: 10px !important; letter-spacing: 0.08em; text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--display) !important; font-size: 24px !important;
    font-weight: 700; color: var(--text) !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

h1, h2, h3 { font-family: var(--display) !important; letter-spacing: -0.02em; }
h1 { font-size: 2rem !important; font-weight: 800 !important; color: var(--text); }
h2 { font-size: 1.25rem !important; font-weight: 700 !important; color: var(--text); }
h3 { font-size: 1rem !important; font-weight: 600 !important; }

hr { border-color: var(--border) !important; margin: 1.5rem 0; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent; border-bottom: 1px solid var(--border); gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted);
    padding: 8px 18px; border-bottom: 2px solid transparent; background: transparent;
}
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

.stButton > button {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: var(--mono) !important; font-size: 12px !important;
    letter-spacing: 0.05em !important; padding: 10px 24px !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: var(--surface2) !important; border-color: var(--border2) !important;
    border-radius: 8px !important; font-size: 13px;
}
.stSlider [data-baseweb="slider"] { padding: 0 2px; }
.stNumberInput input { background: var(--surface2) !important; border-color: var(--border2) !important; border-radius: 8px !important; }

.stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
.stDataFrame thead th {
    background: var(--surface2) !important; font-family: var(--mono) !important;
    font-size: 10px !important; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted) !important;
}

.stInfo { background: rgba(79,124,255,0.08) !important; border-color: rgba(79,124,255,0.3) !important; border-radius: 8px !important; }
.stSuccess { background: rgba(61,255,160,0.08) !important; border-color: rgba(61,255,160,0.3) !important; border-radius: 8px !important; }
.stWarning { background: rgba(255,179,71,0.08) !important; border-color: rgba(255,179,71,0.3) !important; border-radius: 8px !important; }

.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }
.patrol-card {
    background: var(--surface); border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin-bottom: 8px;
    font-size: 13px; font-family: var(--body); line-height: 1.5;
}
.anomaly-card {
    background: rgba(255,77,109,0.06); border-left: 3px solid var(--danger);
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin-bottom: 8px; font-size: 13px;
}
.model-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px; text-align: center; height: 140px;
    display: flex; flex-direction: column; justify-content: center; gap: 4px;
    transition: border-color 0.2s, transform 0.15s;
}
.model-card:hover { border-color: var(--border2); transform: translateY(-2px); }
.model-card .mc-name { font-family: var(--display); font-weight: 700; font-size: 14px; }
.model-card .mc-algo { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.04em; margin-top: 2px; }
.model-card .mc-desc { font-size: 11px; color: #8a9ab8; margin-top: 4px; line-height: 1.4; }
.model-card .mc-loc  { font-family: var(--mono); font-size: 9px; color: var(--muted); margin-top: 6px; }

.page-header { margin-bottom: 2rem; }
.page-header h1 { margin-bottom: 4px; }
.page-caption {
    font-family: var(--mono); font-size: 11px; color: var(--muted);
    letter-spacing: 0.04em; text-transform: uppercase;
}

.badge {
    display: inline-block; font-family: var(--mono); font-size: 9px;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 3px 8px; border-radius: 4px; font-weight: 500;
}
.badge-blue  { background: rgba(79,124,255,0.15); color: #7fa0ff; border: 1px solid rgba(79,124,255,0.25); }
.badge-green { background: rgba(61,255,160,0.12); color: #3dffa0; border: 1px solid rgba(61,255,160,0.2); }
.badge-red   { background: rgba(255,77,109,0.12); color: #ff6b85; border: 1px solid rgba(255,77,109,0.2); }
.badge-amber { background: rgba(255,179,71,0.12); color: #ffb347; border: 1px solid rgba(255,179,71,0.2); }

.sidebar-brand { padding: 12px 0 20px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
.sidebar-brand .sb-title {
    font-family: var(--display); font-size: 16px; font-weight: 800;
    color: var(--text); letter-spacing: -0.02em;
}
.sidebar-brand .sb-sub {
    font-family: var(--mono); font-size: 10px; color: var(--muted);
    letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px;
}
.section-label {
    font-family: var(--mono); font-size: 10px; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; margin-top: 16px;
}

.map-loading-placeholder {
    width: 100%; height: 660px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 12px; color: var(--muted); letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    df     = pd.read_csv(D("clustered_crime.csv"))
    area   = pd.read_csv(D("area_summary.csv"))
    patrol = pd.read_csv(D("patrol_recommendations.csv"))
    return df, area, patrol

@st.cache_data(show_spinner=False)
def load_metrics():
    out = {}
    for fname, key in [("model_metrics.csv",    "model_metrics"),
                        ("per_class_metrics.csv", "per_class"),
                        ("shap_importance.csv",   "shap"),
                        ("confusion_matrix.csv",  "cm")]:
        path = D(fname)
        if os.path.exists(path):
            kw = {"index_col": 0} if key == "cm" else {}
            out[key] = pd.read_csv(path, **kw)
    return out

@st.cache_data(show_spinner=False)
def load_forecast():
    p = D("forecast.csv")
    return pd.read_csv(p, parse_dates=['ds']) if os.path.exists(p) else None

@st.cache_resource(show_spinner=False)
def load_models():
    out = {}
    for fname, key in [("model.pkl",    "XGBoost"),
                        ("rf_model.pkl", "Random Forest"),
                        ("knn_model.pkl","KNN"),
                        ("nb_model.pkl", "Naive Bayes"),
                        ("svm_model.pkl","SVM")]:
        p = M(fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[key] = pickle.load(f)
    with open(M("encoder.pkl"), "rb") as f:
        out["encoder"] = pickle.load(f)
    feat_p = M("features.pkl")
    out["features"] = pickle.load(open(feat_p,"rb")) if os.path.exists(feat_p) else None
    scaler_p = M("scaler.pkl")
    out["scaler"] = pickle.load(open(scaler_p,"rb")) if os.path.exists(scaler_p) else None
    return out

# ── Load ──────────────────────────────────────────────────────────────────────

with st.spinner("Loading..."):
    data, area_summary, patrol = load_data()
    metrics  = load_metrics()
    forecast = load_forecast()
    mdl      = load_models()

encoder  = mdl.get("encoder")
features = mdl.get("features")
scaler   = mdl.get("scaler")
NEEDS_SCALE = {"KNN", "Naive Bayes", "SVM"}
MAP_CENTER  = [data['Latitude'].mean(), data['Longitude'].mean()]

# ── Area names ────────────────────────────────────────────────────────────────

AREA_NAMES = {
    1:"Rogers Park",2:"West Ridge",3:"Uptown",4:"Lincoln Square",
    5:"North Center",6:"Lake View",7:"Lincoln Park",8:"Near North Side",
    9:"Edison Park",10:"Norwood Park",11:"Jefferson Park",12:"Forest Glen",
    13:"North Park",14:"Albany Park",15:"Portage Park",16:"Irving Park",
    17:"Dunning",18:"Montclare",19:"Belmont Cragin",20:"Hermosa",
    21:"Avondale",22:"Logan Square",23:"Humboldt Park",24:"West Town",
    25:"Austin",26:"West Garfield Park",27:"East Garfield Park",28:"Near West Side",
    29:"North Lawndale",30:"South Lawndale",31:"Lower West Side",32:"Loop",
    33:"Near South Side",34:"Armour Square",35:"Douglas",36:"Oakland",
    37:"Fuller Park",38:"Grand Boulevard",39:"Kenwood",40:"Washington Park",
    41:"Hyde Park",42:"Woodlawn",43:"South Shore",44:"Chatham",
    45:"Avalon Park",46:"South Chicago",47:"Burnside",48:"Calumet Heights",
    49:"Roseland",50:"Pullman",51:"South Deering",52:"East Side",
    53:"West Pullman",54:"Riverdale",55:"Hegewisch",56:"Garfield Ridge",
    57:"Archer Heights",58:"Brighton Park",59:"McKinley Park",60:"Bridgeport",
    61:"New City",62:"West Elsdon",63:"Gage Park",64:"Clearing",
    65:"West Lawn",66:"Chicago Lawn",67:"West Englewood",68:"Englewood",
    69:"Greater Grand Crossing",70:"Ashburn",71:"Auburn Gresham",
    72:"Beverly",73:"Washington Heights",74:"Mount Greenwood",
    75:"Morgan Park",76:"O'Hare",77:"Edgewater"
}
def get_area_name(a):
    return AREA_NAMES.get(int(a), f"Area {int(a)}")

def risk_color(score, max_score):
    r = score / max_score if max_score > 0 else 0
    if r > 0.75: return "red"
    if r > 0.50: return "orange"
    if r > 0.25: return "lightred"
    return "green"

# ── Plotly theme helper ───────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter, sans-serif", color="#8a9ab8", size=11),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.04)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.04)'),
    margin=dict(l=10, r=10, t=40, b=10),
    title_font=dict(family="Syne, sans-serif", size=14, color="#e8edf5"),
)

def apply_theme(fig, **overrides):
    layout = {**PLOT_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig

# ── Built-in forecast fallback ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_builtin_forecast(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    daily = df.groupby(df['Date'].dt.date).size().reset_index(name='y')
    daily.columns = ['ds','y']
    daily['ds']      = pd.to_datetime(daily['ds'])
    daily            = daily.sort_values('ds')
    daily['rolling'] = daily['y'].rolling(7, min_periods=1).mean()
    x      = np.arange(len(daily))
    coeffs = np.polyfit(x, daily['y'], 1)
    fx     = np.arange(len(daily), len(daily)+30)
    fdates = [daily['ds'].max() + pd.Timedelta(days=i+1) for i in range(30)]
    fy     = np.polyval(coeffs, fx)
    std    = daily['y'].tail(30).std()
    fdf    = pd.DataFrame({'ds':fdates,'yhat':fy,
                           'yhat_lower':fy-std,'yhat_upper':fy+std})
    return daily, fdf

# ── Map helpers ───────────────────────────────────────────────────────────────

def make_base_map(zoom=11):
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )
    return m

def add_area_labels(m, summary_df, max_risk=None):
    if max_risk is None:
        max_risk = summary_df['Risk_Score'].max()
    for _, row in summary_df.iterrows():
        area_id   = int(row['Community_Area'])
        area_name = get_area_name(area_id)
        color     = risk_color(row['Risk_Score'], max_risk)
        popup_html = f"""
        <div style="font-family:-apple-system,sans-serif;min-width:200px;color:#111">
          <div style="font-weight:700;font-size:14px;margin-bottom:6px">{area_name}</div>
          <table style="font-size:12px;width:100%;border-collapse:collapse">
            <tr><td style="color:#666;padding:2px 8px 2px 0">Area #</td><td><b>{area_id}</b></td></tr>
            <tr><td style="color:#666;padding:2px 8px 2px 0">Total crimes</td><td><b>{int(row['Total_Crimes']):,}</b></td></tr>
            <tr><td style="color:#666;padding:2px 8px 2px 0">Top crime</td><td><b>{row['Most_Common_Crime']}</b></td></tr>
            <tr><td style="color:#666;padding:2px 8px 2px 0">Peak hour</td><td><b>{int(row['Peak_Hour']):02d}:00</b></td></tr>
            <tr><td style="color:#666;padding:2px 8px 2px 0">Risk score</td><td><b>{row['Risk_Score']:.3f}</b></td></tr>
          </table>
        </div>"""
        folium.CircleMarker(
            location=[row['Avg_Latitude'], row['Avg_Longitude']],
            radius=10, color=color, fill=True,
            fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=230),
            tooltip=f"{area_name}  |  Risk {row['Risk_Score']:.3f}"
        ).add_to(m)
        folium.Marker(
            location=[row['Avg_Latitude'], row['Avg_Longitude']],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px;font-weight:700;color:#fff;'
                     f'text-shadow:0 0 4px #000,0 0 4px #000,0 0 4px #000;'
                     f'white-space:nowrap;pointer-events:none;'
                     f'transform:translate(-50%,18px)">{area_name}</div>',
                icon_size=(150,18), icon_anchor=(75,0)
            )
        ).add_to(m)

def add_legend(m):
    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
          background:rgba(6,8,15,0.92);color:#e8edf5;padding:14px 18px;
          border-radius:10px;font-size:12px;font-family:-apple-system,sans-serif;
          line-height:1.8;border:1px solid rgba(255,255,255,0.1)">
      <div style="font-weight:700;font-size:11px;letter-spacing:0.08em;
            text-transform:uppercase;color:#5a6a85;margin-bottom:6px">Risk Level</div>
      <div><span style="color:#e74c3c;font-size:14px">⬤</span>  High (>75%)</div>
      <div><span style="color:#e67e22;font-size:14px">⬤</span>  Medium-high (50–75%)</div>
      <div><span style="color:#f0a070;font-size:14px">⬤</span>  Medium (25–50%)</div>
      <div><span style="color:#2ecc71;font-size:14px">⬤</span>  Low (<25%)</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="sb-title">CrimeLens</div>
  <div class="sb-sub">Chicago · 2012 – 2017</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)

# ── FIX 1: Anomaly Explorer moved before ML Models ────────────────────────────
menu = st.sidebar.radio("", [
    "Executive Dashboard",
    "Area Intelligence",
    "Hotspot Mapping",
    "Crime Prediction",
    "Patrol Planning",
    "Forecast",
    "Anomaly Explorer",
    "ML Models",
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-family:'DM Mono',monospace;font-size:10px;color:#3a4a62;line-height:1.8">
  <div>{len(data):,} incidents</div>
  <div>{data['Crime_Type'].nunique()} crime types</div>
  <div>{data['Community_Area'].nunique()} community areas</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Executive Dashboard
# ══════════════════════════════════════════════════════════════════════════════

if menu == "Executive Dashboard":
    st.markdown('<div class="page-header"><h1>Executive Dashboard</h1><div class="page-caption">City-wide crime analytics — Chicago 2012–2017</div></div>', unsafe_allow_html=True)

    total       = len(data)
    top_crime   = data['Crime_Type'].value_counts().idxmax()
    peak_hour   = int(data['Hour'].mode()[0])
    top_area    = int(area_summary.iloc[0]['Community_Area'])
    n_anomalies = int(data['Is_Anomaly'].sum()) if 'Is_Anomaly' in data.columns else 0
    n_models    = len([k for k in mdl if k not in ("encoder","features","scaler")])

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Incidents",    f"{total:,}")
    c2.metric("Top Crime Type",     top_crime)
    c3.metric("Peak Hour",          f"{peak_hour:02d}:00")
    c4.metric("Highest Risk Area",  f"{get_area_name(top_area)}")
    c5.metric("Anomalies Flagged",  f"{n_anomalies:,}")
    c6.metric("Models Trained",     str(n_models))

    st.markdown("---")

    col_l, col_r = st.columns([1.5,1])
    with col_l:
        st.markdown("#### Crime trend — monthly")
        trend = data.groupby(['Year','Month']).size().reset_index(name='Count')
        trend['Period'] = trend['Year'].astype(str) + "-" + trend['Month'].astype(str).str.zfill(2)
        fig = px.line(trend, x='Period', y='Count', markers=True,
                      color_discrete_sequence=['#4f7cff'])
        fig.update_traces(line_width=2, marker_size=4)
        apply_theme(fig, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Top 10 crime types")
        ct = data['Crime_Type'].value_counts().head(10).reset_index()
        ct.columns = ['Crime_Type','Count']
        fig2 = px.bar(ct, x='Count', y='Crime_Type', orientation='h',
                      color='Count', color_continuous_scale=[[0,'#1a2d5e'],[1,'#4f7cff']])
        fig2.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, coloraxis_showscale=False)
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Top high-risk community areas")
    disp = area_summary.head(10).copy()
    disp['Area Name'] = disp['Community_Area'].apply(get_area_name)
    cols = ['Community_Area','Area Name','Total_Crimes','Most_Common_Crime','Peak_Hour','Risk_Score']
    if 'Anomaly_Count' in disp.columns: cols.insert(5,'Anomaly_Count')
    st.dataframe(disp[cols].rename(columns={
        'Community_Area':'Area #','Total_Crimes':'Crimes','Most_Common_Crime':'Top Crime',
        'Peak_Hour':'Peak Hr','Risk_Score':'Risk','Anomaly_Count':'Anomalies'
    }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Area Intelligence
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "Area Intelligence":
    st.markdown('<div class="page-header"><h1>Area Intelligence</h1><div class="page-caption">Per-community area breakdown</div></div>', unsafe_allow_html=True)

    selected_area = st.selectbox(
        "Community Area",
        options=sorted(area_summary['Community_Area'].unique()),
        format_func=lambda a: f"{get_area_name(a)}  (#{int(a)})"
    )
    area_data = data[data['Community_Area'] == selected_area]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Crimes",  len(area_data))
    c2.metric("Most Common",   area_data['Crime_Type'].mode().iloc[0] if not area_data.empty else "N/A")
    c3.metric("Peak Hour",     f"{int(area_data['Hour'].mode().iloc[0]):02d}:00" if not area_data.empty else "N/A")
    if 'Is_Anomaly' in area_data.columns:
        c4.metric("Anomalies", int(area_data['Is_Anomaly'].sum()))

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Crime Types", "Hourly Pattern", "Day of Week"])

    with tab1:
        vc = area_data['Crime_Type'].value_counts().head(12).reset_index()
        vc.columns = ['Crime_Type','Count']
        fig = px.bar(vc, x='Crime_Type', y='Count',
                     color='Count', color_continuous_scale=[[0,'#4d1020'],[1,'#ff4d6d']],
                     title=f"Crime distribution — {get_area_name(selected_area)}")
        fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        hourly = area_data.groupby('Hour').size().reset_index(name='Count')
        fig2 = px.area(hourly, x='Hour', y='Count', markers=True,
                       color_discrete_sequence=['#ffb347'])
        fig2.update_traces(fillcolor='rgba(255,179,71,0.12)', line_width=2)
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        dow  = area_data.groupby('DayOfWeek').size().reset_index(name='Count')
        dow['Day'] = dow['DayOfWeek'].map(dict(enumerate(days)))
        fig3 = px.bar(dow, x='Day', y='Count',
                      color='Count', color_continuous_scale=[[0,'#1a1040'],[1,'#8b5cf6']])
        fig3.update_layout(coloraxis_showscale=False)
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Hotspot Mapping
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "Hotspot Mapping":
    st.markdown('<div class="page-header"><h1>Hotspot Mapping</h1><div class="page-caption">Click circles for area details · heatmap intensity = crime density</div></div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns([2,1,1])
    with col_f1:
        crime_filter = st.multiselect("Filter by crime type (empty = all)", options=sorted(data['Crime_Type'].unique()), default=[])
    with col_f2:
        heat_sample = st.slider("Heatmap points", 2_000, 20_000, 10_000, step=1_000)
    with col_f3:
        show_labels = st.checkbox("Show area names", value=True)

    filtered = data if not crime_filter else data[data['Crime_Type'].isin(crime_filter)]

    with st.spinner("Rendering map…"):
        m = make_base_map(zoom=11)

        area_crime_count = data.groupby('Community_Area').size().rename('area_count').reset_index()
        max_count = area_crime_count['area_count'].max()
        area_crime_count['weight'] = (
            (area_crime_count['area_count'] / max_count).clip(0, 1) * 0.8 + 0.2
        )

        heat_pts_df = (filtered[['Latitude','Longitude','Community_Area']]
                       .dropna()
                       .sample(min(heat_sample, len(filtered)), random_state=42)
                       .merge(area_crime_count[['Community_Area','weight']], on='Community_Area', how='left'))
        heat_pts_df['weight'] = heat_pts_df['weight'].fillna(0.3)

        HeatMap(
            heat_pts_df[['Latitude','Longitude','weight']].values.tolist(),
            radius=18, blur=25, min_opacity=0.25, max_zoom=14,
            gradient={
                0.0:  '#0a0a2e',
                0.15: '#003f7f',
                0.30: '#0080a0',
                0.45: '#00bfa5',
                0.60: '#7ecf00',
                0.75: '#ffcc00',
                0.88: '#ff6600',
                1.0:  '#cc0022',
            }
        ).add_to(m)

        if show_labels:
            add_area_labels(m, area_summary)

        add_legend(m)

    map_key = f"hotspot_map_{hash(tuple(crime_filter))}_{heat_sample}_{show_labels}"
    st_folium(m, width="100%", height=660, key=map_key, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Crime Prediction
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "Crime Prediction":
    st.markdown('<div class="page-header"><h1>Crime Prediction</h1><div class="page-caption">Predict crime type using trained classifiers</div></div>', unsafe_allow_html=True)

    available_models = [k for k in ["XGBoost","Random Forest","KNN","Naive Bayes","SVM"] if k in mdl]
    model_choice = st.selectbox("Model", available_models)

    algo_info = {
        "XGBoost":       ("Gradient boosted trees. Highest accuracy, handles class imbalance well. Scale-invariant.", "Primary · best F1 score"),
        "Random Forest": ("Ensemble of 150 decision trees. Robust and interpretable. Scale-invariant.", "Ensemble comparison"),
        "KNN":           ("k=15 nearest neighbours by Euclidean distance. Sensitive to scale — uses StandardScaler.", "Distance-based"),
        "Naive Bayes":   ("Probabilistic model assuming feature independence. Fast but lower accuracy on correlated features.", "Probabilistic baseline"),
        "SVM":           ("LinearSVC with Platt calibration. Full RBF-kernel SVM skipped (O(n^2) on 40k rows). Uses StandardScaler.", "Linear SVM"),
    }
    info, where = algo_info.get(model_choice, ("",""))
    st.info(f"**{model_choice}** — {info}  \nUsed for: {where}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Time & Location")
        year           = st.number_input("Year", 2012, 2030, 2019)
        month          = st.slider("Month", 1, 12, 6)
        day            = st.slider("Day", 1, 31, 15)
        hour           = st.slider("Hour", 0, 23, 14)
        dayofweek      = st.slider("Day of week  (0 = Mon, 6 = Sun)", 0, 6, 2)
        is_weekend     = int(dayofweek >= 5)
        community_area = st.number_input("Community Area", 1, 80, 25)
        st.caption(f"Selected: **{get_area_name(community_area)}**")
    with col2:
        st.markdown("##### Coordinates & Context")
        lat      = st.number_input("Latitude",  value=41.8781, format="%.4f")
        lon      = st.number_input("Longitude", value=-87.6298, format="%.4f")
        domestic = st.selectbox("Domestic incident?", [0,1], index=0, format_func=lambda x: "Yes" if x else "No")

    if st.button("Run Prediction", type="primary"):
        h_sin = np.sin(2*np.pi*hour/24);  h_cos = np.cos(2*np.pi*hour/24)
        m_sin = np.sin(2*np.pi*month/12); m_cos = np.cos(2*np.pi*month/12)
        row_d = {
            'Year':year,'Month':month,'Day':day,'Hour':hour,
            'DayOfWeek':dayofweek,'IsWeekend':is_weekend,
            'Hour_Sin':h_sin,'Hour_Cos':h_cos,
            'Month_Sin':m_sin,'Month_Cos':m_cos,
            'Latitude':lat,'Longitude':lon,
            'Community_Area':community_area,
            'Domestic':domestic
        }
        feat_order = [f for f in (features or list(row_d.keys())) if f in row_d]
        X_pred = pd.DataFrame([[row_d[f] for f in feat_order]], columns=feat_order)
        chosen_model = mdl[model_choice]
        X_input = scaler.transform(X_pred) if (model_choice in NEEDS_SCALE and scaler) else X_pred.values
        pred     = chosen_model.predict(X_input)
        proba    = chosen_model.predict_proba(X_input)[0]
        top5_idx = np.argsort(proba)[::-1][:5]
        crime    = encoder.inverse_transform(pred)[0]

        st.success(f"**Prediction ({model_choice}):** {crime}")
        st.caption(f"{get_area_name(community_area)}  ·  {hour:02d}:00  ·  {'Weekend' if is_weekend else 'Weekday'}")

        prob_df = pd.DataFrame({
            'Crime Type':  encoder.inverse_transform(top5_idx),
            'Probability': proba[top5_idx]
        })
        fig = px.bar(prob_df, x='Probability', y='Crime Type', orientation='h',
                     color='Probability',
                     color_continuous_scale=[[0,'#1a2d5e'],[1,'#4f7cff']],
                     title="Top-5 predicted probabilities")
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — Patrol Planning
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "Patrol Planning":
    st.markdown('<div class="page-header"><h1>Patrol Planning</h1><div class="page-caption">Risk-scored recommendations — click map markers for details</div></div>', unsafe_allow_html=True)

    col_rec, col_tbl = st.columns([1, 1.4])
    with col_rec:
        st.markdown("#### Recommendations")
        for _, row in patrol.iterrows():
            css = "anomaly-card" if "[" in str(row.get('Recommendation','')) else "patrol-card"
            st.markdown(f'<div class="{css}">{row["Recommendation"]}</div>', unsafe_allow_html=True)

    with col_tbl:
        st.markdown("#### Priority zones")
        dp = patrol.copy()
        dp['Area Name'] = dp['Community_Area'].apply(get_area_name)
        cols = ['Community_Area','Area Name','Total_Crimes','Most_Common_Crime','Peak_Hour','Risk_Score']
        if 'Anomaly_Count' in dp.columns: cols.insert(4,'Anomaly_Count')
        st.dataframe(dp[cols].rename(columns={
            'Community_Area':'Area #','Total_Crimes':'Crimes',
            'Most_Common_Crime':'Top Crime','Peak_Hour':'Peak Hr',
            'Risk_Score':'Risk','Anomaly_Count':'Anomalies'
        }), use_container_width=True, hide_index=True)

    st.markdown("#### Patrol zone map")
    with st.spinner("Rendering patrol map…"):
        pm = make_base_map()
        max_risk = patrol['Risk_Score'].max()
        for _, row in patrol.iterrows():
            area_id   = int(row['Community_Area'])
            area_name = get_area_name(area_id)
            color     = risk_color(row['Risk_Score'], max_risk)
            popup_html = f"""
            <div style="font-family:-apple-system,sans-serif;min-width:210px;color:#111">
              <div style="font-weight:700;font-size:14px;margin-bottom:6px">{area_name} (#{area_id})</div>
              <div style="font-size:11px;margin-bottom:8px;color:#444">{row.get('Recommendation','')}</div>
              <table style="font-size:11px;width:100%;border-collapse:collapse">
                <tr><td style="color:#666;padding:2px 8px 2px 0">Total crimes</td><td><b>{int(row['Total_Crimes']):,}</b></td></tr>
                <tr><td style="color:#666;padding:2px 8px 2px 0">Risk score</td><td><b>{row['Risk_Score']:.3f}</b></td></tr>
              </table>
            </div>"""
            folium.Marker(
                location=[row['Avg_Latitude'], row['Avg_Longitude']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{area_name}  |  Risk {row['Risk_Score']:.2f}",
                icon=folium.Icon(color=color, icon='shield', prefix='fa')
            ).add_to(pm)
            folium.Marker(
                location=[row['Avg_Latitude'], row['Avg_Longitude']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:9px;font-weight:700;color:#fff;'
                         f'text-shadow:0 0 4px #000,0 0 4px #000;white-space:nowrap;'
                         f'pointer-events:none;transform:translate(-50%,28px)">{area_name}</div>',
                    icon_size=(140,18), icon_anchor=(70,0)
                )
            ).add_to(pm)

        zone_path = D("patrol_zone_centers.csv")
        if os.path.exists(zone_path):
            for _, c in pd.read_csv(zone_path).iterrows():
                folium.CircleMarker(
                    location=[c['Lat'],c['Lon']], radius=18,
                    color='#00e5cc', fill=True, fill_color='#00e5cc',
                    fill_opacity=0.15, weight=1.5,
                    tooltip=f"Patrol Zone {int(c['Zone'])}"
                ).add_to(pm)

        add_legend(pm)

    st_folium(pm, width="100%", height=640, key="patrol_map_v1", returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — Forecast
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "Forecast":
    st.markdown('<div class="page-header"><h1>Crime Volume Forecast</h1><div class="page-caption">Historical trend + 30-day projection</div></div>', unsafe_allow_html=True)

    if forecast is not None:
        st.caption("Prophet seasonal forecast (30-day horizon).")
        cutoff = forecast['ds'].max() - pd.Timedelta(days=30)
        hist   = forecast[forecast['ds'] <= cutoff]
        fcast  = forecast[forecast['ds'] > cutoff]
        hx,hy  = hist['ds'], hist['yhat']
        fx,fy  = fcast['ds'], fcast['yhat']
        flo,fhi = fcast['yhat_lower'], fcast['yhat_upper']
        tbl    = fcast[['ds','yhat','yhat_lower','yhat_upper']].copy()
    else:
        st.caption("Statistical forecast — 7-day rolling average + linear trend. Install Prophet for seasonal accuracy.")
        daily, fcast = build_builtin_forecast(data)
        hx,hy  = daily['ds'], daily['rolling']
        fx,fy  = fcast['ds'], fcast['yhat']
        flo,fhi = fcast['yhat_lower'], fcast['yhat_upper']
        tbl    = fcast.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hx, y=hy, name='Historical (smoothed)',
                             line=dict(color='#4f7cff', width=1.5)))
    fig.add_trace(go.Scatter(
        x=pd.concat([fx, fx[::-1]]),
        y=pd.concat([fhi, flo[::-1]]),
        fill='toself', fillcolor='rgba(255,179,71,0.1)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence band', showlegend=True))
    fig.add_trace(go.Scatter(x=fx, y=fy, name='30-day forecast',
                             line=dict(color='#ffb347', dash='dash', width=2)))
    fig.update_layout(
        title="Daily crime volume — historical + 30-day forecast",
        xaxis_title="Date", yaxis_title="Daily incidents",
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8a9ab8'))
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    col_tbl, col_spark = st.columns([1.2, 1])
    with col_tbl:
        st.markdown("#### Forecast table — next 30 days")
        tbl.columns = ['Date','Predicted','Lower','Upper']
        for c in ['Predicted','Lower','Upper']:
            tbl[c] = tbl[c].round(1)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    with col_spark:
        st.markdown("#### Top-5 crime types — monthly trend")
        dc = data.copy()
        dc['Date'] = pd.to_datetime(dc['Date'], errors='coerce')
        top5 = dc['Crime_Type'].value_counts().head(5).index.tolist()
        monthly = (dc[dc['Crime_Type'].isin(top5)]
                   .groupby([dc['Date'].dt.to_period('M').astype(str),'Crime_Type'])
                   .size().reset_index(name='Count'))
        monthly.columns = ['Period','Crime_Type','Count']
        fig2 = px.line(monthly, x='Period', y='Count', color='Crime_Type', markers=False,
                       color_discrete_sequence=['#4f7cff','#00e5cc','#ffb347','#ff4d6d','#8b5cf6'])
        fig2.update_traces(line_width=1.5)
        apply_theme(fig2, xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — Anomaly Explorer  (moved before ML Models)
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "Anomaly Explorer":
    st.markdown('<div class="page-header"><h1>Anomaly Explorer</h1><div class="page-caption">Incidents flagged by Isolation Forest — 5% contamination rate</div></div>', unsafe_allow_html=True)

    if 'Is_Anomaly' not in data.columns:
        st.warning("Anomaly data not found — re-run model.py.")
    else:
        anomalies = data[data['Is_Anomaly'] == 1]
        st.metric("Anomalous incidents", f"{len(anomalies):,}")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### By crime type")
            ac = anomalies['Crime_Type'].value_counts().head(10).reset_index()
            ac.columns = ['Crime_Type','Count']
            fig = px.pie(ac, values='Count', names='Crime_Type',
                         color_discrete_sequence=[
                             '#4f7cff','#00e5cc','#ffb347','#ff4d6d','#8b5cf6',
                             '#06b6d4','#f97316','#ef4444','#3dffa0','#e879f9'])
            fig.update_traces(textfont_size=11)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### By hour of day")
            ah = anomalies.groupby('Hour').size().reset_index(name='Count')
            fig2 = px.bar(ah, x='Hour', y='Count',
                          color='Count',
                          color_continuous_scale=[[0,'#3a0012'],[1,'#ff4d6d']])
            fig2.update_layout(coloraxis_showscale=False)
            apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        # ── FIX 2: heading is now INSIDE the spinner so it never appears before the map ──
        with st.spinner("Rendering anomaly map…"):
            st.markdown("#### Anomaly map")
            am = make_base_map()
            anom_s = (anomalies.dropna(subset=['Latitude','Longitude'])
                      .sample(min(3000, len(anomalies)), random_state=0))
            mc = MarkerCluster()
            for _, row in anom_s.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5, color='#ff4d6d', fill=True, fill_opacity=0.7,
                    tooltip=f"{row['Crime_Type']}  |  {get_area_name(row['Community_Area'])}  |  {int(row['Hour']):02d}:00"
                ).add_to(mc)
            mc.add_to(am)

            if 'Anomaly_Count' in area_summary.columns:
                for _, row in area_summary.nlargest(10,'Anomaly_Count').iterrows():
                    area_name = get_area_name(row['Community_Area'])
                    folium.Marker(
                        location=[row['Avg_Latitude'], row['Avg_Longitude']],
                        icon=folium.DivIcon(
                            html=f'<div style="background:rgba(255,77,109,0.88);'
                                 f'color:#fff;font-size:9px;font-weight:700;'
                                 f'padding:3px 8px;border-radius:5px;white-space:nowrap;'
                                 f'transform:translate(-50%,-50%)">'
                                 f'{area_name}<br>{int(row["Anomaly_Count"])} anomalies</div>',
                            icon_size=(150,32), icon_anchor=(75,16)
                        )
                    ).add_to(am)

        st_folium(am, width="100%", height=620, key="anomaly_map_v1", returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — ML Models  (moved after Anomaly Explorer)
# ══════════════════════════════════════════════════════════════════════════════

elif menu == "ML Models":
    st.markdown('<div class="page-header"><h1>Machine Learning Models</h1><div class="page-caption">All classifiers and clustering algorithms — performance & explainability</div></div>', unsafe_allow_html=True)

    st.markdown("#### Classifiers")
    algo_cards = [
        ("XGBoost",         "model.py §1", "Gradient boosted trees",
         "Primary · scale-invariant · best F1",          "#4f7cff"),
        ("Random Forest",   "model.py §2", "Bagging ensemble",
         "Robust ensemble · scale-invariant",            "#00e5cc"),
        ("KNN  k=15",       "model.py §3", "k-nearest neighbours",
         "Distance-based · StandardScaler",              "#ffb347"),
        ("Naive Bayes",     "model.py §4", "GaussianNB probabilistic",
         "Fast baseline · feature independence",         "#8b5cf6"),
        ("SVM  LinearSVC",  "model.py §5", "Linear support vector machine",
         "Full RBF-SVM skipped (O(n²)) · scaled",       "#ff4d6d"),
    ]
    cols_a = st.columns(5)
    for i, (name, loc, algo, desc, color) in enumerate(algo_cards):
        with cols_a[i]:
            st.markdown(
                f'<div class="model-card" style="border-top:3px solid {color}">'
                f'<div class="mc-name">{name}</div>'
                f'<div class="mc-algo">{algo}</div>'
                f'<div class="mc-desc">{desc}</div>'
                f'<div class="mc-loc">{loc}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Clustering & Anomaly Detection")
    cluster_cards = [
        ("DBSCAN",           "model.py §6", "Density-based spatial clustering",
         "Hotspot detection · arbitrary-shape clusters", "#06b6d4"),
        ("KMeans  k=10",     "model.py §7", "Centroid-based clustering",
         "Patrol zone assignment · 10 zones",            "#f97316"),
        ("Isolation Forest", "model.py §8", "Anomaly detection",
         "Flags ~5% most unusual incidents",             "#ef4444"),
    ]
    cols_b = st.columns(3)
    for i, (name, loc, algo, desc, color) in enumerate(cluster_cards):
        with cols_b[i]:
            st.markdown(
                f'<div class="model-card" style="border-top:3px solid {color}">'
                f'<div class="mc-name">{name}</div>'
                f'<div class="mc-algo">{algo}</div>'
                f'<div class="mc-desc">{desc}</div>'
                f'<div class="mc-loc">{loc}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    if 'model_metrics' in metrics:
        st.markdown("#### Classifier comparison")
        mm   = metrics['model_metrics']
        best = mm.loc[mm['F1_Weighted'].idxmax()]
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Best model",   best['Model'])
        b2.metric("Accuracy",     f"{best['Accuracy']:.3f}")
        b3.metric("Weighted F1",  f"{best['F1_Weighted']:.3f}")
        if pd.notna(best.get('ROC_AUC')):
            b4.metric("ROC-AUC", f"{best['ROC_AUC']:.3f}")

        fig = px.bar(
            mm.melt(id_vars='Model', value_vars=['Accuracy','F1_Weighted','F1_Macro']),
            x='variable', y='value', color='Model', barmode='group',
            title="Accuracy / F1-Weighted / F1-Macro across all classifiers",
            color_discrete_sequence=['#4f7cff','#00e5cc','#ffb347','#8b5cf6','#ff4d6d'],
            labels={'variable':'Metric','value':'Score'}
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mm.round(4), use_container_width=True, hide_index=True)

    if 'shap' in metrics:
        st.markdown("---")
        st.markdown("#### Feature importance — SHAP (XGBoost)")
        shap_df = metrics['shap'].sort_values('Mean_SHAP', ascending=True).tail(13)
        fig2 = px.bar(shap_df, x='Mean_SHAP', y='Feature', orientation='h',
                      color='Mean_SHAP',
                      color_continuous_scale=[[0,'#0a2a3a'],[1,'#00e5cc']],
                      title="Mean |SHAP| value per feature")
        fig2.update_layout(coloraxis_showscale=False)
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    if 'per_class' in metrics:
        st.markdown("---")
        st.markdown("#### Per-class F1 — XGBoost")
        pc = metrics['per_class'].sort_values('F1', ascending=False)
        fig3 = px.bar(pc, x='Crime_Type', y='F1',
                      color='F1', color_continuous_scale=[[0,'#5a0015'],[0.5,'#ffb347'],[1,'#3dffa0']],
                      title="Per-class F1 score")
        fig3.add_hline(y=0.5, line_dash="dot",
                       annotation_text="0.5 threshold", line_color="#5a6a85")
        fig3.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    if 'cm' in metrics:
        st.markdown("---")
        st.markdown("#### Confusion matrix — XGBoost (top-15 crime types)")
        cm = metrics['cm']
        fig4 = px.imshow(
            cm.values, x=list(cm.columns), y=list(cm.index),
            color_continuous_scale=[[0,'#06080f'],[0.3,'#1a2d5e'],[1,'#4f7cff']],
            aspect='auto', title="Predicted vs Actual"
        )
        apply_theme(fig4, xaxis_tickangle=-35)
        st.plotly_chart(fig4, use_container_width=True)
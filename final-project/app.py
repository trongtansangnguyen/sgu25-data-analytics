import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG & PATHS
# ============================================================================
st.set_page_config(page_title="EAP Economic Dashboard", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "east_asia_pacific"

# Data sources mapping - tên file và cột dữ liệu tương ứng
INDICATORS = {
    "gdp": {
        "label": "GDP",
        "file": "gdp_eap_2000_2024.csv",
        "column": "gdp",
        "unit": "USD (current)",
        "source": "World Bank",
    },
    "cpi": {
        "label": "CPI",
        "file": "cpi_eap_2000_2024.csv",
        "column": "cpi",
        "unit": "Index (2010=100)",
        "source": "World Bank/IMF",
    },
    "pce": {
        "label": "PCE",
        "file": "pce_eap_2000_2024.csv",
        "column": "pce",
        "unit": "USD (current)",
        "source": "World Bank",
    },
    "pop": {
        "label": "Population",
        "file": "pop_eap_2000_2024.csv",
        "column": "population",
        "unit": "Persons",
        "source": "World Bank",
    },
    "gini": {
        "label": "GINI Index",
        "file": "gini_index_eap_2000_2024.csv",
        "column": "gini_index",
        "unit": "Index (0-100)",
        "source": "World Bank",
    },
    "poverty": {
        "label": "Poverty Headcount",
        "file": "poverty_headcount_eap_2000_2024.csv",
        "column": "poverty_365",
        "unit": "% of population",
        "source": "World Bank",
    },
}

TRANSFORMS = {
    "raw": "Raw Values",
    "per_capita": "Per Capita",
    "yoy": "YoY % Change",
    "cagr": "CAGR %",
}

# Default countries to select
DEFAULT_COUNTRIES = ["VNM", "THA", "IDN", "MYS", "PHL"]

# ============================================================================
# DATA LOADING & CACHING
# ============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_indicator(key: str) -> pd.DataFrame | None:
    """Load single indicator with error handling."""
    try:
        if key not in INDICATORS:
            return None
        
        meta = INDICATORS[key]
        file_path = DATA_DIR / meta["file"]
        
        if not file_path.exists():
            return None
        
        df = pd.read_csv(file_path)
        
        # Ensure country_code exists
        if "country_code" not in df.columns:
            return None
        
        # Handle wide format (GINI: country_code + year columns)
        if "year" not in df.columns:
            # Identify year columns (numeric strings like '2000', '2001', etc)
            year_cols = [c for c in df.columns if isinstance(c, (int, str)) and str(c).isdigit()]
            
            if year_cols:
                # Wide format - melt it
                df = df.set_index("country_code")
                df = df[[str(c) for c in sorted(year_cols)]].copy()
                df = df.melt(ignore_index=False, var_name="year", value_name="value")
                df = df.reset_index()
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
            else:
                return None
        else:
            # Long format - extract value column
            value_col = meta["column"]
            
            # If exact column doesn't exist, try first numeric column
            if value_col not in df.columns:
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if numeric_cols:
                    value_col = numeric_cols[0]
                else:
                    return None
            
            # Keep only needed columns
            df = df[["country_code", "year", value_col]].copy()
            df = df.rename(columns={value_col: "value"})
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
        
        # Final cleanup
        df = df.dropna(subset=["value", "year"])
        df["year"] = df["year"].astype(int)
        df["indicator"] = key
        df["unit"] = meta["unit"]
        
        return df[["country_code", "year", "value", "indicator", "unit"]].copy()
    
    except Exception as e:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_all_indicators() -> pd.DataFrame:
    """Load all indicators efficiently."""
    frames = []
    for key in INDICATORS.keys():
        df = load_indicator(key)
        if df is not None:
            frames.append(df)
    
    if not frames:
        st.error("No data loaded!")
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# DATA TRANSFORMATIONS
# ============================================================================

def normalize_by_country(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Normalize values to [0, 1] range by country."""
    df = df.copy()
    grouped = df.groupby("country_code")[value_col]
    df[value_col] = grouped.transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    return df


def apply_per_capita(df: pd.DataFrame, inds: list) -> pd.DataFrame:
    """Calculate per-capita values."""
    df_pop = df[df["indicator"] == "pop"][["country_code", "year", "value"]].rename(
        columns={"value": "pop_value"}
    )
    
    frames = []
    for ind in inds:
        if ind == "pop":
            continue
        sub = df[df["indicator"] == ind].merge(df_pop, on=["country_code", "year"], how="left")
        sub["value"] = sub["value"] / sub["pop_value"]
        sub["series"] = f"{ind}_per_capita"
        frames.append(sub[["country_code", "year", "value", "indicator", "series"]])
    
    keep = df[~df["indicator"].isin(inds)].copy()
    keep["series"] = keep["indicator"]
    
    return pd.concat([keep] + frames, ignore_index=True) if frames else keep


def apply_yoy(df: pd.DataFrame, inds: list) -> pd.DataFrame:
    """Calculate year-over-year percentage change."""
    def calc_yoy(grp):
        grp = grp.sort_values("year").copy()
        grp["value"] = grp["value"].pct_change() * 100
        grp["series"] = f"{grp.iloc[0]['indicator']}_yoy"
        return grp
    
    frames = [
        df[df["indicator"] == ind].groupby("country_code", group_keys=False).apply(calc_yoy)
        for ind in inds if ind != "pop"
    ]
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_cagr(df: pd.DataFrame, inds: list) -> pd.DataFrame:
    """Calculate Compound Annual Growth Rate."""
    def calc_cagr(grp):
        grp = grp.sort_values("year").copy()
        start_year = grp["year"].min()
        start_val = grp.loc[grp["year"] == start_year, "value"].iloc[0]
        grp["series"] = f"{grp.iloc[0]['indicator']}_cagr"
        
        if start_val <= 0:
            grp["value"] = np.nan
        else:
            grp["value"] = grp.apply(
                lambda row: ((row["value"] / start_val) ** (1 / max(row["year"] - start_year, 1)) - 1) * 100
                if row["year"] > start_year else 0,
                axis=1,
            )
        return grp
    
    frames = [
        df[df["indicator"] == ind].groupby("country_code", group_keys=False).apply(calc_cagr)
        for ind in inds if ind != "pop"
    ]
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_transform(df: pd.DataFrame, inds: list, transform: str) -> pd.DataFrame:
    """Apply data transformation."""
    if transform == "raw":
        df_filtered = df[df["indicator"].isin(inds)].copy()
        df_filtered["series"] = df_filtered["indicator"]
        return df_filtered
    
    inds_no_pop = [i for i in inds if i != "pop"]
    
    if transform == "per_capita":
        return apply_per_capita(df, inds)
    elif transform == "yoy":
        return apply_yoy(df, inds_no_pop)
    elif transform == "cagr":
        return apply_cagr(df, inds_no_pop)
    
    # Default to raw
    df_filtered = df[df["indicator"].isin(inds)].copy()
    df_filtered["series"] = df_filtered["indicator"]
    return df_filtered


# ============================================================================
# UI & FILTERING
# ============================================================================

def build_sidebar(df: pd.DataFrame):
    """Build sidebar filters."""
    st.sidebar.title("Filters")
    
    countries = sorted(df["country_code"].unique())
    default_countries = [c for c in DEFAULT_COUNTRIES if c in countries]
    
    selected_countries = st.sidebar.multiselect(
        "Countries",
        countries,
        default=default_countries,
        help="Select countries to visualize"
    )
    
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    selected_years = st.sidebar.slider(
        "Year Range",
        year_min, year_max,
        (max(year_min, year_max - 10), year_max),
        help="Select time period"
    )
    
    indicator_options = list(INDICATORS.keys())
    selected_indicators = st.sidebar.multiselect(
        "Indicators",
        indicator_options,
        default=["gdp", "pop"],
        help="Select indicators to display"
    )
    
    transform = st.sidebar.selectbox(
        "Transform",
        list(TRANSFORMS.keys()),
        format_func=TRANSFORMS.get,
        help="Data transformation method"
    )
    
    return selected_countries, selected_years, selected_indicators, transform


def filter_data(df: pd.DataFrame, countries: list, years: tuple, indicators: list) -> pd.DataFrame:
    """Filter data by countries, years, and indicators."""
    start_year, end_year = years
    mask = (
        df["country_code"].isin(countries)
        & df["indicator"].isin(indicators)
        & df["year"].between(start_year, end_year)
    )
    return df.loc[mask].copy().dropna(subset=["value"])


# ============================================================================
# VISUALIZATION SECTIONS
# ============================================================================

def section_overview(df: pd.DataFrame):
    """Overview metrics."""
    st.subheader("Overview")
    
    if df.empty:
        st.info("Select at least one country and indicator.")
        return
    
    latest_year = int(df["year"].max())
    latest = df[df["year"] == latest_year].groupby("indicator")["value"].mean()
    
    cols = st.columns(min(4, len(latest)))
    for idx, (indicator, value) in enumerate(latest.items()):
        cols[idx % len(cols)].metric(
            INDICATORS.get(indicator, {}).get("label", indicator),
            f"{value:,.2f}",
            help=f"{latest_year} average"
        )


def section_timeseries(df: pd.DataFrame):
    """Time series visualization."""
    st.subheader("Time Series")
    
    if df.empty:
        st.info("No data available for selected filters.")
        return
    
    fig = px.line(
        df,
        x="year",
        y="value",
        color="country_code",
        facet_col="indicator",
        facet_col_wrap=2,
        markers=True,
        hover_data={"value": ":.3f"},
        labels={"value": "Value", "year": "Year", "country_code": "Country"}
    )
    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def section_comparison(df: pd.DataFrame):
    """Comparison by year."""
    st.subheader("Comparison (selected year)")
    
    if df.empty:
        st.info("No data available.")
        return
    
    year_choice = st.slider(
        "Select year",
        int(df["year"].min()),
        int(df["year"].max()),
        int(df["year"].max()),
        key="comp_year"
    )
    
    snap = df[df["year"] == year_choice]
    if snap.empty:
        st.warning(f"No data for {year_choice}.")
        return
    
    fig = px.bar(
        snap,
        x="country_code",
        y="value",
        color="indicator",
        barmode="group",
        hover_data={"value": ":.3f"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def section_heatmap(df: pd.DataFrame):
    """Heatmap of countries vs indicators."""
    st.subheader("Heatmap (latest year)")
    
    if df.empty:
        st.info("No data available.")
        return
    
    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year]
    
    pivot = latest.pivot_table(
        index="country_code",
        columns="indicator",
        values="value",
        aggfunc="mean"
    )
    
    if pivot.empty:
        st.warning("Cannot create heatmap with current selection.")
        return
    
    # Normalize for better heatmap visualization
    pivot_norm = pivot.copy()
    for col in pivot_norm.columns:
        min_val, max_val = pivot_norm[col].min(), pivot_norm[col].max()
        if max_val > min_val:
            pivot_norm[col] = (pivot_norm[col] - min_val) / (max_val - min_val)
    
    fig = px.imshow(
        pivot_norm,
        labels=dict(x="Indicator", y="Country", color="Normalized Value"),
        color_continuous_scale="YlOrRd",
        aspect="auto"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def section_ranking(df: pd.DataFrame):
    """Top-N countries ranking."""
    st.subheader("Ranking")
    
    if df.empty:
        st.info("No data available.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    year_choice = col1.slider(
        "Year",
        int(df["year"].min()),
        int(df["year"].max()),
        int(df["year"].max()),
        key="rank_year"
    )
    
    indicators = sorted(df["indicator"].unique())
    indicator_choice = col2.selectbox("Indicator", indicators, key="rank_ind")
    
    top_n = col3.slider("Top N", 3, 20, 10, key="rank_n")
    
    snap = df[(df["year"] == year_choice) & (df["indicator"] == indicator_choice)]
    if snap.empty:
        st.warning(f"No data for {year_choice} and {indicator_choice}.")
        return
    
    snap = snap.nlargest(top_n, "value")
    
    fig = px.bar(
        snap,
        x="value",
        y="country_code",
        orientation="h",
        color="value",
        color_continuous_scale="Blues",
        hover_data={"value": ":.3f"}
    )
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def section_correlation(df: pd.DataFrame):
    """Correlation matrix."""
    st.subheader("Correlation Analysis")
    
    if df.empty:
        st.info("No data available.")
        return
    
    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year]
    
    pivot = latest.pivot_table(
        index="country_code",
        columns="indicator",
        values="value",
        aggfunc="mean"
    ).dropna(axis=1, how="all")
    
    if pivot.shape[1] < 2:
        st.info("Need at least 2 indicators for correlation analysis.")
        return
    
    corr = pivot.corr(method="pearson")
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


def section_scatter(df: pd.DataFrame):
    """Scatter plot with animation."""
    st.subheader("Scatter Analysis")
    
    if df.empty:
        st.info("No data available.")
        return
    
    indicators = sorted(df["indicator"].unique())
    if len(indicators) < 2:
        st.info("Need at least 2 indicators for scatter plot.")
        return
    
    col1, col2 = st.columns(2)
    x_ind = col1.selectbox("X-axis", indicators, index=0, key="scatter_x")
    y_ind = col2.selectbox("Y-axis", indicators, index=min(1, len(indicators)-1), key="scatter_y")
    
    # Pivot data
    x_data = df[df["indicator"] == x_ind][["country_code", "year", "value"]].rename(columns={"value": f"{x_ind}_val"})
    y_data = df[df["indicator"] == y_ind][["country_code", "year", "value"]].rename(columns={"value": f"{y_ind}_val"})
    
    merged = x_data.merge(y_data, on=["country_code", "year"], how="inner")
    
    if merged.empty:
        st.warning("No overlapping data for selected indicators.")
        return
    
    fig = px.scatter(
        merged,
        x=f"{x_ind}_val",
        y=f"{y_ind}_val",
        color="country_code",
        animation_frame="year",
        hover_name="country_code",
        labels={
            f"{x_ind}_val": f"{INDICATORS[x_ind]['label']}",
            f"{y_ind}_val": f"{INDICATORS[y_ind]['label']}"
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def section_distribution(df: pd.DataFrame):
    """Distribution visualization."""
    st.subheader("Distribution")
    
    if df.empty:
        st.info("No data available.")
        return
    
    col1, col2 = st.columns(2)
    
    year_choice = col1.slider(
        "Year",
        int(df["year"].min()),
        int(df["year"].max()),
        int(df["year"].max()),
        key="dist_year"
    )
    
    chart_type = col2.radio("Chart type", ["box", "violin"], horizontal=True, key="dist_type")
    
    snap = df[df["year"] == year_choice]
    if snap.empty:
        st.warning(f"No data for {year_choice}.")
        return
    
    if chart_type == "box":
        fig = px.box(snap, x="indicator", y="value", color="indicator", points="outliers")
    else:
        fig = px.violin(snap, x="indicator", y="value", color="indicator", box=True, points="all")
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def section_data_info(df_all: pd.DataFrame):
    """Data availability and sources."""
    st.subheader("Data Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Available Indicators:**")
        for ind_key, meta in INDICATORS.items():
            st.caption(f"• {meta['label']}: {meta['source']}")
    
    with col2:
        st.write("**Data Coverage:**")
        countries = df_all["country_code"].nunique()
        years_range = f"{int(df_all['year'].min())}-{int(df_all['year'].max())}"
        st.caption(f"{countries} countries")
        st.caption(f"Years: {years_range}")
    
    st.divider()
    
    # Data availability heatmap
    st.write("**Data Availability Heatmap:**")
    latest_year = int(df_all["year"].max())
    availability = df_all[df_all["year"] == latest_year].pivot_table(
        index="country_code",
        columns="indicator",
        values="value",
        aggfunc="count"
    ).fillna(0).astype(int)
    
    fig = px.imshow(
        availability,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(color="Available")
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def section_download(df: pd.DataFrame):
    """Download filtered data."""
    st.subheader("Download")
    
    if df.empty:
        st.info("No data to download.")
        return
    
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_bytes,
        file_name="eap_economic_data.csv",
        mime="text/csv",
    )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application."""
    st.title("East Asia & Pacific Economic Dashboard")
    st.markdown("Real-time economic indicators for 25+ countries in the EAP region")
    
    # Load all data
    df_all = load_all_indicators()
    
    if df_all.empty:
        st.error("Failed to load data. Please check data files.")
        return
    
    # Sidebar filters
    selected_countries, selected_years, selected_indicators, transform = build_sidebar(df_all)
    
    if not selected_countries or not selected_indicators:
        st.warning("Please select at least one country and one indicator.")
        return
    
    # Filter and transform data
    df_filtered = filter_data(df_all, selected_countries, selected_years, selected_indicators)
    
    if df_filtered.empty:
        st.error("No data available for selected filters.")
        return
    
    df_transformed = apply_transform(df_filtered, selected_indicators, transform)
    
    # Display tabs
    tabs = st.tabs([
        "Overview",
        "Time Series",
        "Comparison",
        "Heatmap",
        "Ranking",
        "Correlation",
        "Scatter",
        "Distribution",
        "Data Info",
        "Download"
    ])
    
    with tabs[0]:
        section_overview(df_transformed)
    with tabs[1]:
        section_timeseries(df_transformed)
    with tabs[2]:
        section_comparison(df_transformed)
    with tabs[3]:
        section_heatmap(df_transformed)
    with tabs[4]:
        section_ranking(df_transformed)
    with tabs[5]:
        section_correlation(df_transformed)
    with tabs[6]:
        section_scatter(df_filtered)
    with tabs[7]:
        section_distribution(df_transformed)
    with tabs[8]:
        section_data_info(df_all)
    with tabs[9]:
        section_download(df_transformed)


if __name__ == "__main__":
    main()

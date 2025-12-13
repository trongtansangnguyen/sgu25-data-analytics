import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "east_asia_pacific"

INDICATORS = {
    "gdp": {
        "label": "GDP (normalized)",
        "path": DATA_DIR / "gdp_eap_processed.csv",
    },
    "cpi": {
        "label": "CPI (normalized)",
        "path": DATA_DIR / "cpi_eap_processed.csv",
    },
    "pce": {
        "label": "PCE (normalized)",
        "path": DATA_DIR / "pce_eap_processed.csv",
    },
    "pop": {
        "label": "Population (normalized)",
        "path": DATA_DIR / "population_eap_processed.csv",
    },
}

SOURCES = {
    "gdp": {
        "name": "Gross Domestic Product (GDP)",
        "source": "World Bank",
        "unit": "USD (current, normalized 0–1)",
        "description": "GDP đo lường tổng giá trị hàng hoá và dịch vụ được sản xuất trong một quốc gia.",
        "years": "2000–2025",
        "coverage": "25 quốc gia EAP chính (AUS, CHN, FJI, IDN, JPN, KOR, THA, VNM, MYS, PHL, SG, v.v.)",
    },
    "cpi": {
        "name": "Consumer Price Index (CPI)",
        "source": "World Bank / IMF",
        "unit": "Index (2010=100, normalized 0–1)",
        "description": "CPI theo dõi sự thay đổi giá các hàng hoá và dịch vụ tiêu dùng; dùng để đo lạm phát.",
        "years": "2000–2024",
        "coverage": "25 quốc gia EAP",
    },
    "pce": {
        "name": "Personal Consumption Expenditure (PCE)",
        "source": "World Bank",
        "unit": "USD (current, normalized 0–1)",
        "description": "PCE đo lường tổng chi tiêu tiêu dùng; phản ánh sức mua và nhu cầu hàng hoá.",
        "years": "2000–2024",
        "coverage": "25 quốc gia EAP",
    },
    "pop": {
        "name": "Population (Total)",
        "source": "World Bank",
        "unit": "Người (normalized 0–1)",
        "description": "Dân số tổng cộng của quốc gia tại đầu năm.",
        "years": "2000–2025",
        "coverage": "25 quốc gia EAP",
    },
}

TRANSFORMS = {
    "raw": "Raw (normalized)",
    "per_capita": "Per-capita (gdp/pop, pce/pop)",
    "yoy": "YoY % change",
    "cagr": "CAGR % from range start",
}


@st.cache_data(show_spinner=False)
def load_indicator(key: str) -> pd.DataFrame:
    meta = INDICATORS[key]
    df = pd.read_csv(meta["path"])
    df = df[df["data_level"] == "national"].copy()
    df = df.melt(
        id_vars=["country_code", "data_level"],
        var_name="year",
        value_name="value",
    )
    df["year"] = df["year"].astype(int)
    df.dropna(subset=["value"], inplace=True)
    df["indicator"] = key
    return df


@st.cache_data(show_spinner=False)
def load_all() -> pd.DataFrame:
    frames = [load_indicator(key) for key in INDICATORS]
    return pd.concat(frames, ignore_index=True)


def filter_base(df: pd.DataFrame, countries, years) -> pd.DataFrame:
    start_year, end_year = years
    mask = df["country_code"].isin(countries) & df["year"].between(start_year, end_year)
    return df.loc[mask].copy()


def _add_series_label(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    df = df.copy()
    df["series"] = df["indicator"] if suffix == "" else df["indicator"] + suffix
    return df


def _per_capita(df: pd.DataFrame, target_inds):
    pop = (
        df[df["indicator"] == "pop"][["country_code", "year", "value"]]
        .rename(columns={"value": "pop_value"})
    )
    frames = []
    for ind in target_inds:
        sub = df[df["indicator"] == ind].merge(pop, on=["country_code", "year"], how="left")
        sub["value"] = sub["value"] / sub["pop_value"]
        sub["series"] = f"{ind}_per_capita"
        frames.append(sub)
    keep = df[~df["indicator"].isin(target_inds)].copy()
    keep["series"] = keep["indicator"]
    return pd.concat([keep] + frames, ignore_index=True)


def _yoy(df: pd.DataFrame, target_inds):
    def pct_change(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("year").copy()
        group["value"] = group["value"].pct_change() * 100
        group["series"] = f"{group.iloc[0]['indicator']}_yoy%"
        return group

    parts = []
    for ind in target_inds:
        g = df[df["indicator"] == ind].groupby("country_code", group_keys=False).apply(pct_change)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _cagr(df: pd.DataFrame, target_inds):
    def calc_cagr(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("year").copy()
        start_year = group["year"].min()
        start_val = group.loc[group["year"] == start_year, "value"].iloc[0]
        group["series"] = f"{group.iloc[0]['indicator']}_cagr%"
        if start_val <= 0:
            group["value"] = np.nan
            return group
        group["value"] = group.apply(
            lambda row: ((row["value"] / start_val) ** (1 / (row["year"] - start_year)) - 1) * 100
            if row["year"] > start_year else 0,
            axis=1,
        )
        return group

    parts = []
    for ind in target_inds:
        g = df[df["indicator"] == ind].groupby("country_code", group_keys=False).apply(calc_cagr)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def apply_transform(df: pd.DataFrame, target_inds, transform: str) -> pd.DataFrame:
    if transform == "raw":
        return _add_series_label(df[df["indicator"].isin(target_inds)], "")

    inds = [ind for ind in target_inds if ind != "pop"]
    if not inds:
        return _add_series_label(df[df["indicator"].isin(target_inds)], "")

    if transform == "per_capita":
        return _per_capita(df[df["indicator"].isin(set(inds + ["pop"]))], inds)
    if transform == "yoy":
        return _yoy(df[df["indicator"].isin(inds)], inds)
    if transform == "cagr":
        return _cagr(df[df["indicator"].isin(inds)], inds)

    return _add_series_label(df[df["indicator"].isin(target_inds)], "")


def build_sidebar(df: pd.DataFrame):
    st.sidebar.title("Filters")
    countries = sorted(df["country_code"].unique())
    default_countries = [c for c in ["VNM", "THA", "IDN", "MYS"] if c in countries]
    selected_countries = st.sidebar.multiselect(
        "Countries (EAP)", countries, default=default_countries
    )

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    selected_years = st.sidebar.slider(
        "Year range", min_value=year_min, max_value=year_max, value=(2005, year_max)
    )

    indicator_options = list(INDICATORS.keys())
    selected_indicators = st.sidebar.multiselect(
        "Indicators", indicator_options, default=indicator_options
    )

    transform = st.sidebar.selectbox("Transform", list(TRANSFORMS.keys()), format_func=TRANSFORMS.get)

    chart_indicator_for_facets = st.sidebar.selectbox(
        "Indicator for sparklines", indicator_options, index=indicator_options.index("gdp") if "gdp" in indicator_options else 0
    )

    return selected_countries, selected_years, selected_indicators, transform, chart_indicator_for_facets


def filter_data(df: pd.DataFrame, countries, years, indicators) -> pd.DataFrame:
    start_year, end_year = years
    mask = (
        df["country_code"].isin(countries)
        & df["indicator"].isin(indicators)
        & df["year"].between(start_year, end_year)
    )
    return df.loc[mask].copy()


def format_series_label(series: str) -> str:
    if series in INDICATORS:
        return INDICATORS[series]["label"]
    return series.replace("_", " ").title()


def overview_section(df: pd.DataFrame):
    st.subheader("Overview")
    if df.empty:
        st.info("Chọn ít nhất một quốc gia và một chỉ số.")
        return

    latest_year = int(df["year"].max())
    latest = df[df["year"] == latest_year]
    series_list = sorted(latest["series"].unique())
    cols = st.columns(min(4, max(1, len(series_list))))
    for idx, series in enumerate(series_list):
        metric_df = latest[latest["series"] == series]
        avg_val = metric_df["value"].mean()
        cols[idx % len(cols)].metric(
            format_series_label(series), f"{avg_val:.3f}", help=f"{latest_year} average"
        )


def timeseries_section(df: pd.DataFrame):
    st.subheader("Time Series")
    if df.empty:
        st.info("Không có dữ liệu trong bộ lọc hiện tại.")
        return
    fig = px.line(
        df,
        x="year",
        y="value",
        color="country_code",
        line_dash="series",
        markers=True,
        hover_data={"series": True, "value":":.3f"},
    )
    fig.update_layout(height=420, legend_title="Country / Series")
    st.plotly_chart(fig, use_container_width=True)


def comparison_section(df: pd.DataFrame):
    st.subheader("Comparison (selected year)")
    if df.empty:
        st.info("Không có dữ liệu để so sánh.")
        return
    year_choice = st.slider(
        "Chọn năm cho biểu đồ cột", int(df["year"].min()), int(df["year"].max()), int(df["year"].max())
    )
    snap = df[df["year"] == year_choice]
    if snap.empty:
        st.warning("Không có dữ liệu cho năm đã chọn.")
        return
    fig = px.bar(
        snap,
        x="country_code",
        y="value",
        color="series",
        barmode="group",
        hover_data={"series": True, "value":":.3f"},
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def correlation_section(df: pd.DataFrame):
    st.subheader("Correlations")
    if df.empty:
        st.info("Không có dữ liệu để tính tương quan.")
        return
    pivot = (
        df.groupby(["country_code", "series"])["value"]
        .mean()
        .unstack("series")
        .dropna(axis=1, how="all")
    )
    if pivot.shape[1] < 2:
        st.info("Cần ít nhất 2 chỉ số để tính tương quan.")
        return
    corr = pivot.corr(method="pearson")
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)


def distribution_section(df: pd.DataFrame):
    st.subheader("Distribution (box / violin)")
    if df.empty:
        st.info("Không có dữ liệu để hiển thị phân bố.")
        return
    year_choice = st.slider(
        "Năm cho phân bố", int(df["year"].min()), int(df["year"].max()), int(df["year"].max()), key="dist_year"
    )
    snap = df[df["year"] == year_choice]
    if snap.empty:
        st.warning("Không có dữ liệu cho năm đã chọn.")
        return
    chart_type = st.radio("Chọn kiểu biểu đồ", ["box", "violin"], horizontal=True, key="dist_chart")
    if chart_type == "box":
        fig = px.box(snap, x="series", y="value", color="series", points="outliers", hover_data={"value":":.3f"})
    else:
        fig = px.violin(
            snap,
            x="series",
            y="value",
            color="series",
            box=True,
            points="all",
            hover_data={"value":":.3f"},
        )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def ranking_section(df: pd.DataFrame):
    st.subheader("Ranking / Top-N")
    if df.empty:
        st.info("Không có dữ liệu để xếp hạng.")
        return
    year_choice = st.slider(
        "Năm cho xếp hạng", int(df["year"].min()), int(df["year"].max()), int(df["year"].max()), key="rank_year"
    )
    series_list = sorted(df["series"].unique())
    series_choice = st.selectbox("Chọn series", series_list, key="rank_series")
    top_n = st.slider("Top N", 3, 15, 5, key="rank_topn")
    snap = df[(df["year"] == year_choice) & (df["series"] == series_choice)]
    if snap.empty:
        st.warning("Không có dữ liệu cho lựa chọn.")
        return
    snap = snap.sort_values("value", ascending=False).head(top_n)
    fig = px.bar(
        snap,
        x="value",
        y="country_code",
        orientation="h",
        color="value",
        color_continuous_scale="Blues",
        hover_data={"value":":.3f"},
    )
    fig.update_layout(height=420, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def facet_lines_section(df: pd.DataFrame):
    st.subheader("Small multiples (faceted lines)")
    if df.empty:
        st.info("Không có dữ liệu để vẽ faceted line.")
        return
    fig = px.line(
        df,
        x="year",
        y="value",
        color="country_code",
        facet_col="series",
        facet_col_wrap=2,
        markers=True,
        hover_data={"value":":.3f"},
    )
    fig.update_layout(height=600)
    fig.for_each_annotation(lambda a: a.update(text=format_series_label(a.text.replace("series=", ""))))
    st.plotly_chart(fig, use_container_width=True)


def scatter_section(base_df: pd.DataFrame):
    st.subheader("GDP vs PCE (size=Population, animation by year)")
    needed = base_df[base_df["indicator"].isin(["gdp", "pce", "pop"])]
    wide = needed.pivot_table(index=["country_code", "year"], columns="indicator", values="value").reset_index()
    if wide.empty or {"gdp", "pce"}.difference(wide.columns):
        st.info("Cần GDP và PCE để vẽ scatter.")
        return
    wide["pop"] = wide.get("pop", 1)
    fig = px.scatter(
        wide,
        x="gdp",
        y="pce",
        color="country_code",
        size="pop",
        animation_frame="year",
        hover_name="country_code",
        labels={"gdp": "GDP", "pce": "PCE"},
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)


def slope_section(df: pd.DataFrame):
    st.subheader("Diverging change between two years")
    if df.empty:
        st.info("Không có dữ liệu để tính chênh lệch.")
        return
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    start_year, end_year = st.slider(
        "Chọn năm bắt đầu/kết thúc", min_year, max_year, (max(min_year, max_year - 5), max_year), key="slope_years"
    )
    start = df[df["year"] == start_year].set_index(["country_code", "series"])["value"]
    end = df[df["year"] == end_year].set_index(["country_code", "series"])["value"]
    merged = pd.concat([start.rename("start"), end.rename("end")], axis=1, join="inner").dropna()
    if merged.empty:
        st.warning("Thiếu dữ liệu cho hai mốc năm.")
        return
    merged["delta"] = merged["end"] - merged["start"]
    merged = merged.reset_index()
    fig = px.bar(
        merged,
        x="country_code",
        y="delta",
        color="series",
        hover_data={"start":":.3f", "end":":.3f", "delta":":.3f"},
    )
    fig.update_layout(height=420, title=f"Change {start_year} → {end_year}")
    st.plotly_chart(fig, use_container_width=True)


def radar_section(df: pd.DataFrame):
    st.subheader("Radar profile")
    if df.empty:
        st.info("Không có dữ liệu để vẽ radar.")
        return
    latest_year = int(df["year"].max())
    year_choice = st.slider("Năm cho radar", int(df["year"].min()), latest_year, latest_year, key="radar_year")
    country_choice = st.selectbox("Quốc gia", sorted(df["country_code"].unique()), key="radar_country")
    snap = df[(df["country_code"] == country_choice) & (df["year"] == year_choice)]
    if snap.empty:
        st.warning("Không có dữ liệu cho lựa chọn.")
        return
    fig = px.line_polar(snap, r="value", theta="series", line_close=True, markers=True)
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def sparkline_section(df: pd.DataFrame, indicator_key: str):
    st.subheader("Country sparklines")
    if df.empty:
        st.info("Không có dữ liệu để vẽ sparklines.")
        return
    target_series = [s for s in df["series"].unique() if s.startswith(indicator_key)]
    if not target_series:
        st.info("Không tìm thấy series phù hợp với lựa chọn.")
        return
    sub = df[df["series"].isin(target_series)]
    fig = px.line(
        sub,
        x="year",
        y="value",
        color="country_code",
        facet_col="country_code",
        facet_col_wrap=4,
        hover_data={"value":":.3f"},
    )
    fig.update_layout(height=720, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def missingness_section(base_df: pd.DataFrame):
    st.subheader("Data availability")
    if base_df.empty:
        st.info("Không có dữ liệu để kiểm tra missingness.")
        return
    presence = (
        base_df.pivot_table(index="country_code", columns="year", values="value", aggfunc="size")
        .fillna(0)
    )
    fig = px.imshow(presence, aspect="auto", color_continuous_scale="Greys", origin="lower")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def sources_section():
    st.subheader("Nguồn dữ liệu")
    
    st.markdown("""
    Dashboard này sử dụng dữ liệu **chuẩn hoá (normalized)** từ 4 chỉ số chính của khu vực Đông Á & Thái Bình Dương (EAP):
    """)
    
    for ind_key in ["gdp", "cpi", "pce", "pop"]:
        if ind_key not in SOURCES:
            continue
        src = SOURCES[ind_key]
        with st.expander(f"📊 {src['name']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Nguồn:** {src['source']}")
                st.write(f"**Đơn vị:** {src['unit']}")
            with col2:
                st.write(f"**Khoảng thời gian:** {src['years']}")
                st.write(f"**Phạm vi:** {src['coverage']}")
            st.write(f"**Mô tả:** {src['description']}")
    
    st.divider()
    st.markdown("""
    ### Ghi chú xử lý dữ liệu
    - **Chuẩn hoá (0–1):** Tất cả chỉ số đã được chuẩn hoá về [0, 1] để so sánh công bằng giữa các quốc gia.
    - **Per-capita:** GDP/Pop, PCE/Pop để so sánh trên cơ sở mỗi người.
    - **YoY %:** Tỷ lệ thay đổi năm-trên-năm = (Giá trị năm t / Giá trị năm t-1 - 1) × 100.
    - **CAGR %:** Tỷ lệ tăng trưởng kép hằng năm từ đầu kỳ.
    
    ### Cấu trúc tệp dữ liệu
    - Các file CSV gốc (raw): `data/gdp.csv`, `data/cpi.csv`, `data/pce.csv`, `data/pop.csv`
    - Các file đã xử lý (processed): `data/east_asia_pacific/*_eap_processed.csv`
    - Bộ lọc: chỉ 25 quốc gia EAP được giữ lại để trọng tâm.
    
    ### Cảnh báo
    - Một số quốc gia/năm có thể thiếu dữ liệu (NaN); kiểm tra tab "Availability" để xác nhận.
    - Dữ liệu là snapshot tại thời điểm tạo dashboard; có thể cần cập nhật định kỳ từ nguồn.
    """)


def download_section(df: pd.DataFrame):
    st.subheader("Download subset")
    if df.empty:
        st.info("Không có dữ liệu để tải xuống.")
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Tải CSV đã lọc",
        data=csv_bytes,
        file_name="eap_filtered.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="EAP Dashboard", layout="wide")
    st.title("East Asia & Pacific Indicators (normalized)")

    df = load_all()

    selected_countries, selected_years, selected_indicators, transform, facet_indicator = build_sidebar(df)

    base_filtered = filter_base(df, selected_countries, selected_years)
    transformed = apply_transform(base_filtered, selected_indicators, transform)
    transformed = transformed[
        transformed["indicator"].isin(selected_indicators)
        | transformed["series"].str.contains("per_capita|yoy%|cagr%")
    ].copy()
    transformed.dropna(subset=["value"], inplace=True)

    tab_overview, tab_ts, tab_comp, tab_dist, tab_rank, tab_facets, tab_scatter, tab_change, tab_radar, tab_spark, tab_corr, tab_missing, tab_sources, tab_dl = st.tabs(
        [
            "Overview",
            "Time Series",
            "Comparison",
            "Distribution",
            "Ranking",
            "Small multiples",
            "Scatter",
            "Change",
            "Radar",
            "Sparklines",
            "Correlation",
            "Availability",
            "Data Sources",
            "Download",
        ]
    )

    with tab_overview:
        overview_section(transformed)
    with tab_ts:
        timeseries_section(transformed)
    with tab_comp:
        comparison_section(transformed)
    with tab_dist:
        distribution_section(transformed)
    with tab_rank:
        ranking_section(transformed)
    with tab_facets:
        facet_lines_section(transformed)
    with tab_scatter:
        scatter_section(base_filtered)
    with tab_change:
        slope_section(transformed)
    with tab_radar:
        radar_section(transformed)
    with tab_spark:
        sparkline_section(transformed, facet_indicator)
    with tab_corr:
        correlation_section(transformed)
    with tab_missing:
        missingness_section(base_filtered)
    with tab_sources:
        sources_section()
    with tab_dl:
        download_section(transformed)


if __name__ == "__main__":
    main()

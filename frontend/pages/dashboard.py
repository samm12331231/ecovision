import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import json


import inspect


def _fill_kwargs(fn):
    """Pick the 'fill container width' kwarg that THIS Streamlit function
    actually accepts, so we never pass an unsupported argument.

    Streamlit changed this API three times:
      * >= 1.49           -> width="stretch"
      * ~1.16 .. 1.48     -> use_container_width=True
      * older st.image    -> use_column_width=True
    We inspect the function's real signature and only return a kwarg that
    exists, preferring the newest supported form.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    try:
        mj, mn = (int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        mj, mn = 0, 0
    if (mj, mn) >= (1, 49) and "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    if "use_column_width" in params:
        return {"use_column_width": True}
    return {}


# Resolve once at import time, per widget type (their APIs differ by version).
_IMG_W   = _fill_kwargs(st.image)
_CHART_W = _fill_kwargs(st.plotly_chart)
_BTN_W   = _fill_kwargs(st.download_button)

# ── palette ──────────────────────────────────────────────────────────────────
G_DARK   = "#0d3b2e"
G_MID    = "#1a5c3e"
G_LIGHT  = "#27ae60"
G_BRIGHT = "#2ecc71"
G_PALE   = "#a8e6cf"
AMBER    = "#f39c12"
RED      = "#e74c3c"
BLUE     = "#3498db"
BG_CARD  = "#0f2318"

# Semantic colors for the 4 land-cover classes — kept in sync with the
# backend COLOR_MAP in backend/main.py so the dashboard legend, the
# land-cover donut, and the segmentation overlay all agree.
CLASS_NAMES  = {0: "Vegetation", 1: "Sand", 2: "Water", 3: "Urban"}
CLASS_COLORS = {
    0: "#2ecc71",  # vegetation - green
    1: "#f4d03f",  # sand       - yellow
    2: "#3498db",  # water      - blue
    3: "#e74c3c",  # urban      - red
}

CSS = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(160deg, {G_DARK} 0%, #071a10 100%);
}}
[data-testid="stHeader"] {{ background: transparent; }}

.eco-header {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {G_BRIGHT};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 2px solid {G_MID};
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}}
.metric-card {{
    background: {BG_CARD};
    border: 1px solid {G_MID};
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 4px 18px rgba(0,0,0,0.45);
    margin-bottom: .5rem;
}}
.metric-icon  {{ font-size: 1.9rem; margin-bottom: .2rem; }}
.metric-label {{
    font-size: .72rem;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: {G_PALE};
    margin-bottom: .3rem;
}}
.metric-value {{
    font-size: 2.1rem;
    font-weight: 800;
    color: {G_BRIGHT};
    line-height: 1;
}}
.metric-sub {{ font-size: .7rem; color: #6fcf97; margin-top: .2rem; }}
.badge {{ display: inline-block; padding: .25rem .75rem; border-radius: 999px;
          font-size: .75rem; font-weight: 600; letter-spacing: .05em; }}
.badge-green {{ background: rgba(46,204,113,.18); color: {G_BRIGHT}; border: 1px solid {G_BRIGHT}; }}
.badge-amber {{ background: rgba(243,156,18,.18);  color: {AMBER};    border: 1px solid {AMBER};    }}
.badge-red   {{ background: rgba(231,76,60,.18);   color: {RED};      border: 1px solid {RED};      }}
.eco-divider {{ height: 1px; background: linear-gradient(90deg, transparent, {G_MID}, transparent); margin: 1.2rem 0; }}
.json-box {{
    background: #071a10;
    border: 1px solid {G_MID};
    border-radius: 10px;
    padding: .9rem 1rem;
    font-family: monospace;
    font-size: .72rem;
    color: {G_PALE};
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 220px;
    overflow-y: auto;
}}
</style>
"""


def _card(icon, label, value, sub=""):
    return (
        f'<div class="metric-card">'
        f'<div class="metric-icon">{icon}</div>'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        + (f'<div class="metric-sub">{sub}</div>' if sub else "")
        + "</div>"
    )


def _health_badge(score):
    if score >= 60:
        return '<span class="badge badge-green">&#9679; HEALTHY</span>'
    elif score >= 30:
        return '<span class="badge badge-amber">&#9679; MODERATE</span>'
    return '<span class="badge badge-red">&#9679; STRESSED</span>'


def _gauge(value, title, color=G_BRIGHT):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        title={"text": title, "font": {"size": 13, "color": G_PALE}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": G_MID,
                     "tickfont": {"color": G_PALE, "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": BG_CARD,
            "bordercolor": G_MID,
            "steps": [
                {"range": [0,  30], "color": "#1a0a08"},
                {"range": [30, 60], "color": "#1a1408"},
                {"range": [60, 100], "color": "#0a1a0e"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=G_PALE,
    )
    return fig


def _bar_chart(labels, values, colors, title):
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{int(v):,}" for v in values],
        textposition="outside",
        textfont=dict(color=G_PALE, size=11),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=G_PALE, size=13)),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=G_PALE,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, color=G_PALE),
        yaxis=dict(showgrid=True, gridcolor=G_MID, color=G_PALE),
        showlegend=False,
    )
    return fig


def _donut(labels, values, colors, title):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker_colors=colors,
        textfont_size=11,
        textfont_color=G_PALE,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=G_PALE, size=13)),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=G_PALE,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(font=dict(color=G_PALE, size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def dashboard_section(result, image_data):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<p class="eco-header">&#128202; Analysis Results</p>', unsafe_allow_html=True)

    # ── empty state ──────────────────────────────────────────────────────────
    if result is None:
        st.markdown(
            '<div style="text-align:center; padding:3rem 1rem; color:#3d7a57;">'
            '<div style="font-size:3.5rem;">&#127807;</div>'
            '<div style="font-size:1rem; margin-top:.8rem; letter-spacing:.05em;">'
            'Upload an image and click <b style="color:#2ecc71;">Analyze</b> to see results'
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    # Surface any per-stage failures without hiding the stages that did work
    stage_errors = result.get("errors", {})
    if stage_errors:
        msg = "  •  ".join(f"**{k}**: {v}" for k, v in stage_errors.items())
        st.warning(f"Some stages reported issues — {msg}")

    res = result.get("results", {})

    seg       = res.get("segmentation", {})
    det       = res.get("detection",    {})
    chg       = res.get("change",       {})
    edg       = res.get("edges",        {})

    health    = float(seg.get("health_score",   0))
    n_objects = int(det.get("num_objects",       0))
    changed   = int(chg.get("changed_pixels",   0))
    edges     = int(edg.get("edge_pixels",       0))
    proc_ms   = result.get("processing_time_ms", 0)
    fname     = result.get("filename",           "—")
    img_size  = result.get("image_size",         [0, 0])

    # ── original image ───────────────────────────────────────────────────────
    if image_data:
        img = Image.open(image_data)
        st.image(img, caption="📷 Uploaded Image", **_IMG_W)

    st.markdown('<div class="eco-divider"></div>', unsafe_allow_html=True)

    # ── status row ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.markdown(
            f'<div style="color:{G_PALE}; font-size:.8rem;">Vegetation Health</div>'
            f"{_health_badge(health)}",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="color:{G_PALE}; font-size:.8rem;">&#9201; Processing</div>'
            f'<span style="color:{G_BRIGHT}; font-weight:700;">{proc_ms} ms</span>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div style="color:{G_PALE}; font-size:.8rem;">&#128193; File</div>'
            f'<span style="color:{G_PALE}; font-size:.8rem;">{fname} &nbsp;'
            f"({img_size[0]}x{img_size[1]})</span>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="eco-divider"></div>', unsafe_allow_html=True)

    # ── 4 metric cards ───────────────────────────────────────────────────────
    cc = st.columns(4)
    cards = [
        ("&#127807;", "Veg Health",  f"{health:.1f}%",  "Segmentation", cc[0]),
        ("&#128230;", "Objects",     str(n_objects),     "Detected",     cc[1]),
        ("&#128293;", "Changes",     f"{changed:,}",     "Changed px",   cc[2]),
        ("&#128208;", "Edges",       f"{edges:,}",       "Edge px",      cc[3]),
    ]
    for icon, label, value, sub, col in cards:
        with col:
            st.markdown(_card(icon, label, value, sub), unsafe_allow_html=True)

    st.markdown('<div class="eco-divider"></div>', unsafe_allow_html=True)

    # ── overlays ────────────────────────────────────────────────────────────
    if seg.get("overlay_b64"):
        st.subheader("🗺️ Segmentation Overlay")
        st.image(seg["overlay_b64"], **_IMG_W)
        # Color legend so the overlay is interpretable (matches CLASS_COLORS)
        legend_items = "".join(
            f'<span style="display:inline-flex; align-items:center; margin-right:1rem; font-size:.75rem; color:{G_PALE};">'
            f'<span style="width:12px; height:12px; border-radius:3px; background:{CLASS_COLORS[c]}; '
            f'display:inline-block; margin-right:.35rem;"></span>{CLASS_NAMES[c]}</span>'
            for c in (0, 1, 2, 3)
        )
        st.markdown(
            f'<div style="margin:.2rem 0 .6rem;">{legend_items}</div>',
            unsafe_allow_html=True,
        )

    if chg.get("has_baseline") and chg.get("overlay_b64"):
        st.subheader("🔥 Change Detection")
        st.image(chg["overlay_b64"], **_IMG_W)

    if edg.get("overlay_b64"):
        st.subheader("📐 Edge Detection")
        st.image(edg["overlay_b64"], **_IMG_W)

    st.markdown('<div class="eco-divider"></div>', unsafe_allow_html=True)

    # ── charts row ───────────────────────────────────────────────────────────
    left_col, right_col = st.columns(2)

    with left_col:
        st.plotly_chart(
            _gauge(health, "Vegetation Health %", G_BRIGHT),
            **_CHART_W,
            config={"displayModeBar": False},
        )

    with right_col:
        st.plotly_chart(
            _bar_chart(
                ["Health", "Objects", "Changes", "Edges"],
                [health, n_objects, min(changed, 9999), min(edges, 9999)],
                [G_BRIGHT, BLUE, AMBER, G_PALE],
                "Metric Overview",
            ),
            **_CHART_W,
            config={"displayModeBar": False},
        )

    # ── donut row ────────────────────────────────────────────────────────────
    left2, right2 = st.columns(2)

    with left2:
        has_baseline = chg.get("has_baseline", False)
        total_px     = max(img_size[0] * img_size[1], 1)
        change_pct   = min(changed / total_px * 100, 100)
        stable_pct   = 100 - change_pct
        st.plotly_chart(
            _donut(
                ["Changed", "Stable"],
                [max(change_pct, 0.001), stable_pct],
                [AMBER, G_MID],
                "Change Distribution" + (" (baseline)" if has_baseline else ""),
            ),
            **_CHART_W,
            config={"displayModeBar": False},
        )

    with right2:
        classes_found = seg.get("classes_found", [])
        # Prefer the real per-class proportions from the backend; fall back to an
        # even split only if an older backend didn't send class_distribution.
        dist = seg.get("class_distribution", {})  # {"0": 12.3, "1": 40.1, ...}
        if dist:
            items    = sorted(((int(k), float(v)) for k, v in dist.items()),
                              key=lambda kv: kv[0])
            labels_c = [CLASS_NAMES.get(c, f"Class {c}") for c, _ in items]
            vals_c   = [v for _, v in items]
            colors_c = [CLASS_COLORS.get(c, G_PALE) for c, _ in items]
        elif classes_found:
            labels_c = [CLASS_NAMES.get(c, f"Class {c}") for c in classes_found]
            vals_c   = [100 / len(classes_found)] * len(classes_found)
            # Color each slice by its actual land-cover class, not by position
            colors_c = [CLASS_COLORS.get(c, G_PALE) for c in classes_found]
        else:
            labels_c = vals_c = colors_c = None

        if labels_c:
            st.plotly_chart(
                _donut(labels_c, vals_c, colors_c, "Land-Cover Classes"),
                **_CHART_W,
                config={"displayModeBar": False},
            )
        else:
            st.markdown(
                '<div style="color:#3d7a57; text-align:center; padding:3rem;">No class data</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="eco-divider"></div>', unsafe_allow_html=True)

    # ── export ───────────────────────────────────────────────────────────────
    col_exp, col_json = st.columns([1, 2])
    with col_exp:
        st.download_button(
            "&#128229; Export JSON Report",
            json.dumps(result, indent=2),
            file_name="ecovision_report.json",
            mime="application/json",
            **_BTN_W,
        )
    with col_json:
        with st.expander("&#128269; Raw API Response"):
            st.markdown(
                f'<div class="json-box">{json.dumps(result, indent=2)}</div>',
                unsafe_allow_html=True,
            )

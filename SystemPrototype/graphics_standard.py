import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_image
from docx.shared import Inches


# ==========================================================
# SELECT THE RELEVANT COLUMN OF DATAFRAME
# ==========================================================

def select_relevant_column(df: pd.DataFrame, question: str) -> list:
    """
    Chooses the most relevant numeric column(s) based on the question text.
    Returns a list of column names to plot.
    """
    q = question.lower()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Dual-column comparison (difference, compare, max vs min)
    if "difference" in q or "compare" in q or ("max" in q and "min" in q):
        candidates = [c for c in numeric_cols if any(k in c.lower() for k in ["max", "min"])]
        return candidates[:2] if candidates else numeric_cols[:2]

    # Mean / average
    if "mean" in q or "average" in q:
        col = next((c for c in numeric_cols if "mean" in c.lower()), numeric_cols[0])
        return [col]

    # Maximum / highest
    if "max" in q or "highest" in q:
        col = next((c for c in numeric_cols if "max" in c.lower()), numeric_cols[0])
        return [col]

    # Minimum / lowest
    if "min" in q or "lowest" in q:
        col = next((c for c in numeric_cols if "min" in c.lower()), numeric_cols[0])
        return [col]

    # Variability, distribution, boxplot → all columns
    if any(k in q for k in ["variability", "distribution", "spread", "range", "boxplot"]):
        # Default to 'mean' if exists, otherwise all columns
        mean_col = next((c for c in numeric_cols if "mean" in c.lower()), None)
        return [mean_col] if mean_col else numeric_cols

    # Default to 'mean' if question mentions a topic but not the stat
    if any(k in q for k in ["temperature", "precipitation", "pressure", "rain", "wet", "climate", "weather"]):
        mean_col = next((c for c in numeric_cols if "mean" in c.lower()), None)
        if mean_col:
            return [mean_col]

    # Default fallback: first numeric column
    return [numeric_cols[0]] if numeric_cols else []

# ==========================================================
# SAVE GRAPH TO FILE
# ==========================================================

def save_graph(fig, question):
    """Save Plotly figure as PNG file for Word export."""
    chart_path = f"graph_{abs(hash(question))}.png"
    fig.update_layout(margin=dict(t=100, b=50, l=80, r=80))
    fig.write_image(chart_path, width=1200, height=500, scale=2)
    return chart_path

# ==========================================================
# GENERATE GRAPHICS FOR STANDARD USERS
# ==========================================================
def generate_graphics(doc, chat_history):
    """
    Generates charts right after each Q&A block.
    Uses 3 main chart types: boxplot, multi-bar, and dual-column.
    """
    for entry in chat_history:
        question = entry["question"]
        answer = entry["answer"]
        df = entry.get("numeric_df")

        if df is None or df.empty:
            continue

        # Detect topic
        q_lower = question.lower()

        # Identify columns
        x_col = next((c for c in df.columns if "month" in c.lower()), df.columns[0])
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            continue
        numeric_cols = select_relevant_column(df, question)
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        if not numeric_cols:
            continue

        # BOX PLOT (for median or variability questions)
        if any(k in q_lower for k in ["median", "variability", "range", "spread"]):
            fig = go.Figure()
            for col in numeric_cols:
                fig.add_trace(go.Box(y=df[col], name=col, boxpoints="all", jitter=0.5))
            fig.update_layout(
                title=f"Distribution — {question}",
                yaxis_title="Value",
                boxmode="group"
            )

        # MULTI-BAR CHART
        elif len(numeric_cols) > 1:
            q_lower = question.lower()

            # --- Detect if data includes multiple years ---
            year_col = next((c for c in df.columns if "year" in c.lower() or "anno" in c.lower()), None)

            if year_col:
                x_col = year_col
            else:
                # fallback to Month
                x_col = next((c for c in df.columns if "month" in c.lower()), df.columns[0])

            # --- Flatten all numeric values for global highlight logic ---
            global_max = df[numeric_cols].max().max()
            global_min = df[numeric_cols].min().min()

            highlight_max = any(k in q_lower for k in ["highest", "maximum", "warmest", "wettest", "max"])
            highlight_min = any(k in q_lower for k in ["lowest", "minimum", "coldest", "driest", "min"])

            # --- Create grouped bar chart manually (for flexibility) ---
            fig = go.Figure()
            for col in numeric_cols:
                color_list = ["skyblue"] * len(df)

                if highlight_max:
                    max_idx = df[col].idxmax()
                    if pd.notna(df.loc[max_idx, col]) and np.isclose(df.loc[max_idx, col], global_max, atol=1e-6):
                        color_list[df.index.get_loc(max_idx)] = "crimson"

                elif highlight_min:
                    min_idx = df[col].idxmin()
                    if pd.notna(df.loc[min_idx, col]) and np.isclose(df.loc[min_idx, col], global_min, atol=1e-6):
                        color_list[df.index.get_loc(min_idx)] = "dodgerblue"


                fig.add_trace(
                    go.Bar(
                        x=df[x_col],
                        y=df[col],
                        name=col,
                        marker_color=color_list,
                        text=[f"{v:.2f}" if pd.notna(v) else "" for v in df[col]],
                        textposition="outside"
                    )
                )

            # --- Add global annotation for the extreme ---
            if highlight_max:
                max_row, max_col = None, None
                for c in numeric_cols:
                    idx = df[c].idxmax()
                    if df.loc[idx, c] == global_max:
                        max_row, max_col = idx, c
                        break
                if max_row is not None:
                    fig.add_annotation(
                        x=df.loc[max_row, x_col],
                        y=global_max,
                        text=f"Peak: {global_max:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="crimson",
                        font=dict(color="crimson")
                    )

            elif highlight_min:
                min_row, min_col = None, None
                for c in numeric_cols:
                    idx = df[c].idxmin()
                    if df.loc[idx, c] == global_min:
                        min_row, min_col = idx, c
                        break
                if min_row is not None:
                    fig.add_annotation(
                        x=df.loc[min_row, x_col],
                        y=global_min,
                        text=f"Lowest: {global_min:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="dodgerblue",
                        font=dict(color="dodgerblue")
                    )

            # --- Layout and style ---
            fig.update_layout(
                title=f"{question}",
                barmode="group",
                xaxis_title=x_col,
                yaxis_title="Value",
                title_x=0.5,
                legend_title_text="Metrics",
            )

            if len(df) > 8:
                fig.update_layout(xaxis_tickangle=-45)


        # DUAL-COLUMN CHART (for min/max or comparisons)
        elif any(k in q_lower for k in ["max", "min", "difference", "compare"]):
            # Identify possible comparison columns
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[:2]
                y1, y2 = df[col1].mean(), df[col2].mean()
            else:
                col1, col2 = numeric_cols[0], numeric_cols[0]
                y1, y2 = df[col1].min(), df[col1].max()

            fig = go.Figure(data=[
                go.Bar(x=[col1, col2], y=[y1, y2],
                       marker_color=["steelblue", "crimson"],
                       text=[f"{y1:.2f}", f"{y2:.2f}"],
                       textposition="outside")
            ])
            fig.update_layout(
                title=f"{question}",
                yaxis_title="Value",
                xaxis_title="",
                bargap=0.5,
                title_x=0.5
            )
            diff = y2 - y1
            fig.add_annotation(
                x=0.5, xref="paper",
                y=max(y1, y2),
                text=f"Δ = {diff:.2f}",
                showarrow=False,
                bgcolor="lightyellow",
                font=dict(size=12)
            )

        # Default fallback (single-column bar)
        else:
            y_col = numeric_cols[0]
            fig = px.bar(df, x=x_col, y=y_col, text=y_col,
                         title=f"{question}", labels={y_col: "Value"})
            fig.update_traces(textposition="outside", marker_color="skyblue")

        # --- Save and insert ---
        chart_path = f"graph_{abs(hash(question))}.png"
        fig.write_image(chart_path, width=1200, height=500, scale=2)

        # --- Accessibility ---    
        inline_shape = doc.add_picture(chart_path, width=Inches(6))
        inline = inline_shape._inline
        docPr = inline.docPr
        docPr.set('descr', 'Bar or boxplot chart visually representing the data values')
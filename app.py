
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "altair",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, mo, pd


@app.cell
def _(mo, pd):
    import sys
    from pathlib import Path
    from urllib.request import urlopen

    location = mo.notebook_location()
    data_path = location / "public" / "data" / "penguins.csv"

    if sys.platform == "emscripten":
        with urlopen(str(data_path)) as response:
            df = pd.read_csv(response)
    else:
        df = pd.read_csv(Path(data_path))

    df = df.dropna(
        subset=[
            "species",
            "island",
            "bill_length_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
    )

    return (df,)


@app.cell
def _(mo):
    mo.md("""
    # 🐧 Palmer Penguins Explorer

    Explore penguin measurements across **Adelie**, **Chinstrap**,
    and **Gentoo** species.

    Compare flipper length, body mass, and species averages using
    the interactive charts below.
    """)
    return


@app.cell
def _(df, mo):
    species_options = ["All species"] + sorted(
        df["species"].unique().tolist()
    )

    species = mo.ui.dropdown(
        options=species_options,
        value="All species",
        label="Filter by species",
    )

    species
    return (species,)


@app.cell
def _(df, species):
    filtered_df = (
        df
        if species.value == "All species"
        else df[df["species"] == species.value]
    )
    return (filtered_df,)


@app.cell
def _(filtered_df, mo):
    mo.hstack(
        [
            mo.stat(
                label="Penguins",
                value=f"{len(filtered_df):,}",
            ),
            mo.stat(
                label="Average body mass",
                value=f"{filtered_df['body_mass_g'].mean():,.0f} g",
            ),
            mo.stat(
                label="Average flipper length",
                value=f"{filtered_df['flipper_length_mm'].mean():.1f} mm",
            ),
        ],
        justify="space-around",
    )
    return


@app.cell
def _(alt, filtered_df):
    scatter = (
        alt.Chart(filtered_df)
        .mark_circle(size=100, opacity=0.8)
        .encode(
            x=alt.X(
                "flipper_length_mm:Q",
                title="Flipper length (mm)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "body_mass_g:Q",
                title="Body mass (g)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "species:N",
                title="Species",
                scale=alt.Scale(scheme="set2"),
            ),
            tooltip=[
                alt.Tooltip("species:N", title="Species"),
                alt.Tooltip("island:N", title="Island"),
                alt.Tooltip("sex:N", title="Sex"),
                alt.Tooltip(
                    "flipper_length_mm:Q",
                    title="Flipper length (mm)",
                ),
                alt.Tooltip(
                    "body_mass_g:Q",
                    title="Body mass (g)",
                ),
            ],
        )
        .properties(
            title="Flipper length vs. body mass",
            width="container",
            height=420,
        )
        .interactive()
    )

    scatter
    return


@app.cell
def _(alt, filtered_df):
    summary = (
        filtered_df.groupby("species", as_index=False)
        .agg(
            average_mass=("body_mass_g", "mean"),
            penguin_count=("body_mass_g", "size"),
        )
    )

    bars = (
        alt.Chart(summary)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(
                "species:N",
                title=None,
                sort="-y",
            ),
            y=alt.Y(
                "average_mass:Q",
                title="Average body mass (g)",
            ),
            color=alt.Color(
                "species:N",
                title="Species",
                scale=alt.Scale(scheme="set2"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("species:N", title="Species"),
                alt.Tooltip(
                    "average_mass:Q",
                    title="Average mass (g)",
                    format=".0f",
                ),
                alt.Tooltip(
                    "penguin_count:Q",
                    title="Penguins",
                ),
            ],
        )
        .properties(
            title="Average body mass by species",
            width="container",
            height=320,
        )
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelFontSize=12,
            titleFontSize=14,
        )
        .configure_title(
            fontSize=18,
            anchor="start",
        )
    )

    bars
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Explore the data

    Search, sort, filter, and inspect individual observations
    using marimo's built-in dataframe viewer.
    """)
    return


@app.cell
def _(filtered_df):
    filtered_df
    return


if __name__ == "__main__":
    app.run()

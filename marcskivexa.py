"""This application experiments with the (grid) layout and some styling
Can we make a compact dashboard across several columns and with a dark theme?"""
import io
from typing import List, Optional

import markdown
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from plotly import express as px
from plotly.subplots import make_subplots

# matplotlib.use("TkAgg")
matplotlib.use("Agg")
COLOR = "black"
BACKGROUND_COLOR = "#fff"

DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

def main():
    """Main function. Run this to run the app"""
    st.sidebar.title("Controlling")
    st.markdown(
        """
# Bewegungsdaten verschiedener Datenquellen  - Social Distancing
Resulate von politischen Maßnamen sowie andere Faktoren die sich auf die Anzahl der Infektionen auswirken.
"""
    )

    select_block_container_style()

    # Map with data from uber | EXAMPLE FROM STREAMLIT
    place1 = load_data(100000)

    hour = st.slider("Hour to look at", 0, 23)

    place1 = place1[place1[DATE_TIME].dt.hour == hour]

    st.subheader("Geo data between %i:00 and %i:00" % (hour, (hour + 1) % 24))
    midpoint = (np.average(place1["lat"]), np.average(place1["lon"]))

    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": midpoint[0],
            "longitude": midpoint[1],
            "zoom": 11,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=place1,
                get_position=["lon", "lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ],
    ))

    # My preliminary idea of an API for generating a grid
    with Grid("1 1 1", color=COLOR, background_color=BACKGROUND_COLOR) as grid:
        grid.cell(
            class_="a",
            grid_column_start=2,
            grid_column_end=3,
            grid_row_start=1,
            grid_row_end=2,
        ).markdown("# Hier vielleicht plots oder Tabellen oder einfach nur Text.")
        grid.cell("b", 2, 3, 2, 3).text("The cell to the left is a dataframe")
        grid.cell("c", 3, 4, 2, 3).text("The cell to the left is a textframe")
        grid.cell("d", 1, 2, 1, 3).dataframe(get_dataframe())
        grid.cell("e", 3, 4, 1, 2).markdown(
            "Try changing the **block container style** in the sidebar!"
        )
        grid.cell("f", 1, 3, 3, 4).text(
            "The cell to the right is a matplotlib svg image"
        )
        grid.cell("g", 3, 4, 3, 4).pyplot(get_matplotlib_plt())

    st.plotly_chart(get_plotly_subplots())



class Cell:
    """A Cell can hold text, markdown, plots etc."""

    def __init__(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        self.class_ = class_
        self.grid_column_start = grid_column_start
        self.grid_column_end = grid_column_end
        self.grid_row_start = grid_row_start
        self.grid_row_end = grid_row_end
        self.inner_html = ""

    def _to_style(self) -> str:
        return f"""
.{self.class_} {{
    grid-column-start: {self.grid_column_start};
    grid-column-end: {self.grid_column_end};
    grid-row-start: {self.grid_row_start};
    grid-row-end: {self.grid_row_end};
}}
"""

    def text(self, text: str = ""):
        self.inner_html = text

    def markdown(self, text):
        self.inner_html = markdown.markdown(text)

    def dataframe(self, dataframe: pd.DataFrame):
        self.inner_html = dataframe.to_html()


    def pyplot(self, fig=None, **kwargs):
        string_io = io.StringIO()
        plt.savefig(string_io, format="svg", fig=(2, 2))
        svg = string_io.getvalue()[215:]
        plt.close(fig)
        self.inner_html = '<div height="200px">' + svg + "</div>"

    def _to_html(self):
        return f"""<div class="box {self.class_}">{self.inner_html}</div>"""


class Grid:
    """A (CSS) Grid"""

    def __init__(
        self,
        template_columns="1 1 1",
        gap="10px",
        background_color=COLOR,
        color=BACKGROUND_COLOR,
    ):
        self.template_columns = template_columns
        self.gap = gap
        self.background_color = background_color
        self.color = color
        self.cells: List[Cell] = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        st.markdown(self._get_grid_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_style(), unsafe_allow_html=True)
        st.markdown(self._get_cells_html(), unsafe_allow_html=True)

    def _get_grid_style(self):
        return f"""
<style>
    .wrapper {{
    display: grid;
    grid-template-columns: {self.template_columns};
    grid-gap: {self.gap};
    background-color: {self.background_color};
    color: {self.color};
    }}
    .box {{
    background-color: {self.color};
    color: {self.background_color};
    border-radius: 5px;
    padding: 20px;
    font-size: 150%;
    }}
    table {{
        color: {self.color}
    }}
</style>
"""

    def _get_cells_style(self):
        return (
            "<style>"
            + "\n".join([cell._to_style() for cell in self.cells])
            + "</style>"
        )

    def _get_cells_html(self):
        return (
            '<div class="wrapper">'
            + "\n".join([cell._to_html() for cell in self.cells])
            + "</div>"
        )

    def cell(
        self,
        class_: str = None,
        grid_column_start: Optional[int] = None,
        grid_column_end: Optional[int] = None,
        grid_row_start: Optional[int] = None,
        grid_row_end: Optional[int] = None,
    ):
        cell = Cell(
            class_=class_,
            grid_column_start=grid_column_start,
            grid_column_end=grid_column_end,
            grid_row_start=grid_row_start,
            grid_row_end=grid_row_end,
        )
        self.cells.append(cell)
        return cell


def select_block_container_style():
    """Add selection section for setting setting the max-width and padding
    of the main block container"""
    st.sidebar.header("Block Container Style")
    max_width_100_percent = st.sidebar.checkbox("Max-width: 100%?", False)
    if not max_width_100_percent:
        max_width = st.sidebar.slider("Select max-width in px", 100, 2000, 1200, 100)
    else:
        max_width = 1200
    dark_theme = st.sidebar.checkbox("Dark Theme?", False)
    if dark_theme:
        global COLOR
        global BACKGROUND_COLOR
        BACKGROUND_COLOR = "rgb(17,17,17)"
        COLOR = "#fff"

    _set_block_container_style(
        max_width,
        max_width_100_percent,
    )


def _set_block_container_style(
    max_width: int = 1200,
    max_width_100_percent: bool = False,
    padding_top: int = 5,
    padding_right: int = 1,
    padding_left: int = 1,
    padding_bottom: int = 10,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache
def get_dataframe() -> pd.DataFrame():
    """Dummy DataFrame"""
    data = [
        {"Daten x": 1, "Daten y": 2},
        {"Daten x": 3, "Daten y": 5},
        {"Daten x": 4, "Daten y": 8},
        {"Daten x": 4, "Daten y": 8},
    ]
    return pd.DataFrame(data)


def get_plotly_fig():
    """Dummy Plotly Plot"""
    return px.line(data_frame=get_dataframe(), x="Daten x", y="Daten y")

@st.cache(persist=True)
def load_data(nrows):
    place1 = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    place1.rename(lowercase, axis="columns", inplace=True)
    place1[DATE_TIME] = pd.to_datetime(place1[DATE_TIME])
    return place1

def get_matplotlib_plt():
    get_dataframe().plot(kind="line", x="Daten x", y="Daten y", figsize=(5, 3))


def get_plotly_subplots():
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Table 4"),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
    )

    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)

    fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]), row=1, col=2)

    fig.add_trace(go.Scatter(x=[300, 400, 500], y=[600, 700, 800]), row=2, col=1)

    fig.add_table(
        header=dict(values=["A Scores", "B Scores"]),
        cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]),
        row=2,
        col=2,
    )

    if COLOR == "black":
        template="plotly"
    else:
        template ="plotly_dark"
    fig.update_layout(
        height=500,
        width=700,
        title_text="Weitere Plots und Tabellen",
        template=template,
    )
    return fig


main()

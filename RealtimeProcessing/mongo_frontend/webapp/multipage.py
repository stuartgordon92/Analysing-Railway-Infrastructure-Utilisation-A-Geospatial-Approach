import dash
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash import dcc
import pandas as pd
import plotly.express as px
import random
import os
os.environ['USE_PYGEOS'] = '0'
from shapely.geometry import Point, LineString, MultiLineString
import geopandas as gpd

import pydeck as pdk
import dash_deck
import pymongo


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


MONGO_URL=os.getenv'mongo_URL'
MONGO_URL_FE=os.getenv'mongo_URL_FE'
MONGO_DB=os.getenv'mongo_DB'

RUNNING_TABLE_NAME='history'
HISTORY_TABLE_NAME='history'
UTIL_TABLE_NAME='utilisation'

def mongo(url, db, table):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db = mongo_client[db]
    mongo_tbl = mongo_db[table]
    return mongo_tbl

running_table = mongo(MONGO_URL, MONGO_DB, RUNNING_TABLE_NAME)
history_table = mongo(MONGO_URL_FE, MONGO_DB, HISTORY_TABLE_NAME)
util_table = mongo(MONGO_URL_FE, MONGO_DB, UTIL_TABLE_NAME)

def get_train_summary(mongo, lim=10):
    resp = list(mongo.find({"running_date": 20230627},
                                   {"_id":1,
                                    "CIF_train_uid":1,
                                    "signalling_id":1,
                                    "running_date":1,
                                    "origin_location":1,
                                    "destination_location":1
                                    }))
    
    resp = pd.DataFrame(resp)
    resp = resp.rename(columns={"_id":"id",
                         "CIF_train_uid":"UID",
                         "signalling_id":"Headcode",
                         "running_date":"Running Date",
                         "origin_location":"Origin",
                         "destination_location":"Destination"}).to_dict(orient='records')
    return resp

def get_train_data(mongo, id_list):
    resp = list(mongo.find({'_id':{"$in": id_list}}))
    return resp

summary = get_train_summary(history_table, 1000)


# Set up your Mapbox token (you have to sign up and get this token)
mapbox_api_token = 'pk.eyJ1Ijoic3R1YXJ0Z29yZG9uOTIiLCJhIjoiY2xqMm1kZWt4MHZ4MzNubGQ2amdlMDE0YiJ9.F-hF72o4M0QVeAOS1oNJig'



navbar = dbc.NavbarSimple(
    children=[
        dbc.Button("Sidebar", outline=True, color="secondary", className="mr-1", id="btn_sidebar"),
        dbc.NavItem(dbc.NavLink("Page 1", href="/page-1")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="/page-2"),
                dbc.DropdownMenuItem("Page 3", href="/page-3"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="GeoTRUST",
    brand_href="/",
    color="dark",
    dark=True,
    fluid=True,
)


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        # html.H2("GeoT", className="display-4"),
        # html.Hr(),
        html.P(
            "Click any of the analysis pages below", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Page 1", href="/page-1", id="page-1-link"),
                dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
                dbc.NavLink("Page 3", href="/page-3", id="page-3-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(

    id="page-content",
    style=CONTENT_STYLE)

app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
)


@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        view_state = pdk.ViewState(latitude=54.093409, longitude=-2.89479, zoom=6)
        map_view = pdk.View("MapView", controller=True)
        initial_deck = pdk.Deck(initial_view_state=view_state, views=[map_view])
        initial_deck = initial_deck.to_json()

        page1 = dbc.Container(
            children = [
                html.H1("T2 Geospatial Viewer"),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col([
                            # here we define the deckgl component with id='deckgl'
                            dash_deck.DeckGL(
                                id="deckgl",
                                data=initial_deck,
                                tooltip=True,
                                mapboxKey=mapbox_api_token
                                )
                        ]),
                        dbc.Col([
                            dash_table.DataTable(
                                id='table',
                                columns=[{"name": i, "id": i} for i in list(summary[0].keys())
                                        if i != 'id'],
                                data=summary,
                                fixed_rows={'headers': True},
                                style_as_list_view=True,
                                sort_action="native",
                                filter_action='native',
                                selected_rows=[],
                                row_selectable="multi",
                                style_table={
                                    'height': 400,
                                    'overflowY': 'scroll',
                                    'backgroundColor': 'white'
                                },
                                style_data={
                                    'width': '{}%'.format(100. / len(list(summary[0].keys()))),
                                    'textOverflow': 'hidden',
                                    'backgroundColor': 'white'
                                }
                            ),
                            html.Div(id='table-container', style={'backgroundColor':'white'})
                        ], width=3),
                    ]),
            ],
            fluid=True,
        )
        return page1



        return html.P("This is the content of page 1!")
    elif pathname == "/page-2":
        return html.P("This is the content of page 2. Yay!")
    elif pathname == "/page-3":
        return html.P("Oh cool, this is page 3!")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=True)

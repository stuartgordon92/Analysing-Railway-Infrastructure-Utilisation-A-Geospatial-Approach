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
    print("Getting train summary")
    resp = list(mongo.find({"running_date": 20230710},
                                   {"_id":1,
                                    "CIF_train_uid":1,
                                    "signalling_id":1,
                                    "running_date":1,
                                    "origin_location":1,
                                    "destination_location":1
                                    }).sort("finished_running_timestamp", -1).limit(lim))
    print(f"Train summary: {len(resp)} trains")
    resp = pd.DataFrame(resp)
    resp = resp.rename(columns={"_id":"id",
                         "CIF_train_uid":"UID",
                         "signalling_id":"Headcode",
                         "running_date":"Running Date",
                         "origin_location":"Origin",
                         "destination_location":"Destination"}).to_dict(orient='records')
    return resp

def get_train_data(mongo, id_list):
    print("Getting train data")
    resp = list(mongo.find({'_id':{"$in": id_list}}))
    print(f"Train data: {len(resp)} trains")
    return resp

summary = get_train_summary(history_table, 100)


# Set up your Mapbox token (you have to sign up and get this token)
mapbox_api_token = 'pk.eyJ1Ijoic3R1YXJ0Z29yZG9uOTIiLCJhIjoiY2xqMm1kZWt4MHZ4MzNubGQ2amdlMDE0YiJ9.F-hF72o4M0QVeAOS1oNJig'

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# initialize an empty deck
view_state = pdk.ViewState(latitude=54.093409, longitude=-2.89479, zoom=6)
map_view = pdk.View("MapView", controller=True)
initial_deck = pdk.Deck(initial_view_state=view_state, views=[map_view])
initial_deck = initial_deck.to_json()


app.layout = dbc.Container(
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

@app.callback(
    Output('deckgl', 'data'),
    Input('table', 'derived_virtual_row_ids'),
    Input('table', 'selected_row_ids'))

def update_map(row_ids, selected_row_ids):
    if selected_row_ids is None:
        return dash.no_update
    else:
        selected_train = get_train_data(history_table, selected_row_ids)
        path_layers = []
        print(f"Selected Trains - {len(selected_train)}")
        for ind, selected in enumerate(selected_train):
            geo_data = gpd.read_file(selected['geo_data'], driver='GeoJSON')
            print(f"{selected_row_ids[ind]} No of GeoLocations - {len(geo_data)}")
            if len(geo_data) != 0:
                geo_data['line-color'] = generate_color()
                geo_data['id'] = selected_row_ids[ind]
                geo_data['path'] = geo_data['geometry'].apply(linestring_to_coords)
                geo_data["color"] = geo_data["line-color"].apply(hex_to_rgb)
                geo_data = pd.DataFrame(geo_data[['time', 'event', 'path', 'color', 'id']])
                filtered_data = geo_data.to_dict(orient='records')

                path_layer = pdk.Layer(
                    "PathLayer",
                    data=filtered_data,
                    width_min_pixels=3,
                    pickable=True,

                    get_color='color',
                    auto_highlight=True
                )
                path_layers.append(path_layer)

        r = pdk.Deck(layers=path_layers, initial_view_state=view_state)

        print(f"Selected train: {selected_row_ids}")
        return r.to_json() # we return the dict representation of the deck

def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16)*1 for i in (0, 2, 4)]
def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color
def linestring_to_coords(geom):
    if isinstance(geom, LineString):
        x = list(geom.coords)
        t = [list(coord) for coord in x]
        return  t

    elif isinstance(geom, MultiLineString):
        x = [list(line.coords) for line in geom.geoms]
        t = [[list(coord) for coord in e] for e in x]
        b = [item for sublist in t for item in sublist]
        return b

    else:
        print(geom)
        raise ValueError('Geometry must be a LineString or MultiLineString')
        

if __name__ == '__main__':
    app.run_server(debug=True)

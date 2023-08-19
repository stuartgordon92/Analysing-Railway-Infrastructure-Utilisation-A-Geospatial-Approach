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

from datetime import datetime
import pydeck as pdk
import dash_deck
import pymongo
import math
import pickle
from pydeck.types import String
import copy

MONGO_URL=os.getenv'mongo_URL'
MONGO_URL_FE=os.getenv'mongo_URL_FE'
MONGO_DB=os.getenv'mongo_DB'

RUNNING_TABLE_NAME='running'
HISTORY_TABLE_NAME='history'
UTIL_TABLE_NAME='utilisation'

def mongo(url, db, table):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db = mongo_client[db]
    mongo_tbl = mongo_db[table]
    return mongo_tbl

delay_col = {
    'EARLY': [0, 153, 51],
    'ON TIME': [0, 255, 0],
    'LATE': [255, 0, 0],
    'OFF ROUTE': [255, 204, 0],
}


# data_load_path = 'NetworkModelWGS84_v1.pkl'
# with open(data_load_path, 'rb') as file:
#     # Call load method to deserialze
#     model = pickle.load(file)
# networkmodel_wgs84 = model

# networkmodel_wgs84['color'] = networkmodel_wgs84.apply(lambda x: [217, 217, 217], axis=1)
# networkmodel_wgs84['points'] = networkmodel_wgs84.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
# networkmodel_wgs84 = networkmodel_wgs84[['ASSETID', 'points', 'color']]

# base_layer = pdk.Layer(
#     type="PathLayer",
#     data=networkmodel_wgs84,
#     get_color="color",
#     width_scale=1,
#     width_min_pixels=1,
#     get_path="points",
#     get_width=1,
# )


running_table = mongo(MONGO_URL, MONGO_DB, RUNNING_TABLE_NAME)
history_table = mongo(MONGO_URL_FE, MONGO_DB, HISTORY_TABLE_NAME)
util_table = mongo(MONGO_URL_FE, MONGO_DB, UTIL_TABLE_NAME)

def get_train_realtime(mongo, lim=10):
    date = datetime.today().strftime('%Y%m%d')
    resp = list(mongo.find({"running_date": int(date)},
                                   {"_id":1,
                                    "td_messages":1,
                                    'last_updated_TTL':1,
                                    'signalling_id':1,
                                    'current_variation_status':1,
                                    }).sort("last_updated_TTL", -1).limit(lim))
    print(f"Train summary: {len(resp)} trains")
    resp = pd.DataFrame(resp)
    resp = resp.rename(columns={"_id":"id",
                         "last_updated_TTL":"last_updated"}).to_dict(orient='records')
    print(f"Train summary 2: {len(resp)} trains")
    return resp

# summary = get_train_realtime(running_table, 100)


# Set up your Mapbox token (you have to sign up and get this token)
mapbox_api_token = 'pk.eyJ1Ijoic3R1YXJ0Z29yZG9uOTIiLCJhIjoiY2xqMm1kZWt4MHZ4MzNubGQ2amdlMDE0YiJ9.F-hF72o4M0QVeAOS1oNJig'

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# initialize an empty deck
view_state = pdk.ViewState(latitude=54.093409, longitude=-2.89479, zoom=6)
map_view = pdk.View("MapView", controller=True)


# initial_deck = pdk.Deck(layers=base_layer,initial_view_state=view_state, views=[map_view])
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
                        tooltip={
                                "html": "{signalling_id} - {variation_status}",
                                "style": {
                                        "backgroundColor": "steelblue",
                                        "color": "white"
                                }
                                },
                        mapboxKey=mapbox_api_token
                        ),
                    dcc.Interval(
                            id='interval-component',
                            interval=90 * 1000,
                            n_intervals=0
                        ),
                ])
            ]),
    ],
    fluid=True,
)

@app.callback(
    Output('deckgl', 'data'),
    Input('interval-component', 'n_intervals'))


def update_map(interval):
    print(f"Start Update: {datetime.now()}")

    # path_layers = [base_layer]
    path_layers = []
    # selected_train = get_train_data(history_table, selected_row_ids)
    selected_train = get_train_realtime(running_table,100)
    
    trains=[]
    for i, x in enumerate(selected_train):

        if 'td_messages' in x:
            coordiantes_data = []
            timestamp_data = []
            if isinstance(x['td_messages'], list):
                start_time = x['td_messages'][-1]['time']

                for y in reversed(x['td_messages']):
                    if 'geo' in y:
                        coordiantes_data.append(y['geo']['geometry']['coordinates'])
                        # print(f"{int(y['time'])} - {int((int(start_time) - int(y['time']))/1e3)}")

                        timestamp_data.append(int((int(start_time) - int(y['time']))/1e3))
                        # timestamp_data.append(int(int(y['time'])/1e3))
                details = {
                    'id':x['id'],
                    'signalling_id':x['signalling_id'],
                    'variation_status':x['current_variation_status'],
                    'coordinates': coordiantes_data,
                    'timestamp': timestamp_data,
                    'color': delay_col['ON TIME'] if int(y['timetable_variation']) < 1 else delay_col['LATE']
                }
                trains.append(details)
    train_data = pd.DataFrame(trains)
    print(train_data)
        # filtered_data = geo_data.to_dict(orient='records')

    path_layer = pdk.Layer(
        "TripsLayer",
        data=train_data,
        pickable=True,
        opacity=0.8,
        get_path="coordinates",
        get_timestamps="timestamps",
        get_color='color',
        width_min_pixels=2,
        rounded=True,
        trail_length=60,
        current_time=30,
        # current_time=int(datetime.now().timestamp()),
    )
    print(int(datetime.now().timestamp()))
    path_layers.append(path_layer)

    r = pdk.Deck(layers=path_layers, initial_view_state=view_state)
    
    print(f"Updated: {datetime.now()}, Trains: {i}")
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

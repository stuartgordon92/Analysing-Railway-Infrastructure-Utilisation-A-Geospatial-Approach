import dash
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash import dcc
import pandas as pd
import plotly.express as px
import json
from dynamodb_json import json_util as dbjson
import random
import os
os.environ['USE_PYGEOS'] = '0'
from shapely.geometry import Point, LineString, MultiLineString
from shapely import geometry, ops
import osmnx as ox
import geopandas as gpd
import pickle

import pydeck as pdk
import dash_deck

# train_test_loc = r'C:\Users\stuar\Downloads\extract\qnclfzwub45tpg3lfglzh6gjci.json'
train_test_loc = r'C:\Users\stuar\Downloads\extract\test.json'
# train_test_loc = r'C:\Users\stuar\OneDrive - Newcastle University\Year 1\Thesis\realtime\v5\lambda\geo\test.json'
pickles = r'C:\Users\stuar\OneDrive - Newcastle University\Year 1\Thesis\realtime\v5\train-viewer'

file1 = open(train_test_loc, 'r')
data = {}
data_summary = {}
Lines = file1.readlines()
for line in Lines:
    trains_str = json.loads(line)
    x = dbjson.loads(trains_str['Item'])
    if 'td_messages' in x and len(x['td_messages']) > 2:
        x_id = x['id']
        data[x_id] = x
        data_summary[x_id] = {'UID': x['CIF_train_uid'] if 'CIF_train_uid' in x else x_id[:10],
                            'Headcode': x['signalling_id'] if 'signalling_id' in x else x_id[2:6],
                            'Date': x['running_date'] if 'running_date' in x else x_id[-8:],
                            '# TDs': len(x['td_messages']) if 'td_messages' in x else 0
                        }

data_load_path = os.path.join(pickles, 'utilisation_data_v1.pkl')
with open(data_load_path, 'rb') as file:
    # Call load method to deserialze
    utilisation_data = pickle.load(file)
G, nodes, edges, berths_gdf_folium = utilisation_data

df1 = pd.DataFrame.from_dict(data_summary, orient='index').reset_index(drop=False, names='id')
df2 = pd.DataFrame.from_dict(data, orient='index')


ICON_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/c/c4/Projet_bi%C3%A8re_logo_v2.png"
)
icon_data = {
    # Icon from Wikimedia, used the Creative Commons Attribution-Share Alike 3.0
    # Unported, 2.5 Generic, 2.0 Generic and 1.0 Generic licenses
    "url": ICON_URL,
    "width": 242,
    "height": 242,
    "anchorY": 242,
}

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
                        columns=[{"name": i, "id": i} for i in df1.columns[1:]],
                        data=df1.to_dict('records'),
                        fixed_rows={'headers': True},
                        style_as_list_view=True,
                        sort_action="native",
                        filter_action='native',
                        selected_rows=[],
                        row_selectable="single",
                        style_table={
                            'height': 400,
                            'overflowY': 'scroll',
                            'backgroundColor': 'white'
                        },
                        style_data={
                            'width': '{}%'.format(100. / len(df1.columns)),
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

# @app.callback(
#     Output('deckgl', 'data'),
#     [Input('table', 'active_cell'),
#      Input('table', 'data')]
# )
# def update_map(active_cell, data):

@app.callback(
    Output('deckgl', 'data'),
    Input('table', 'derived_virtual_row_ids'),
    Input('table', 'selected_row_ids'))

def update_map(row_ids, selected_row_ids):
    print ("Initial Selection", selected_row_ids)
    # selected_id_set = set(selected_row_ids or [])
    if selected_row_ids is None:
        return dash.no_update
    else:
        selected_train = df2[df2['id'].isin(selected_row_ids)]

        path_layers = []
        for index, row in selected_train.iterrows():
            geo_data = process_train_data(selected_train)
            geo_data['id'] = row['id']
            geo_data['line-color'] = generate_color()
            geo_data['distance'] = geo_data.apply(lambda row : row['from_coordinates'].distance(row['to_coordinates']) , axis = 1)
            print(list(geo_data.columns))

            geo_points = gpd.GeoDataFrame(pd.concat([geo_data[['time','event','from', 'from_coordinates']].rename(columns={'from': 'berth','from_coordinates': 'geometry' }), 
                                                     geo_data[['time','event','to','to_coordinates']].rename(columns={'to': 'berth','to_coordinates': 'geometry' })],
                                                     ignore_index=True),geometry='geometry').drop_duplicates().astype({'time': 'int64'}).sort_values(by='time').reset_index(drop=True)
            geo_points['lon'] = geo_points['geometry'].x
            geo_points['lat'] = geo_points['geometry'].y
            
            print(geo_points)

            if geo_data['path_geom'].isnull().values.any():
                geo_data = geo_data[geo_data['path_geom'].notna()]

            geo_data = geo_data[['time', 'event', 'path_geom', 'line-color', 'id','distance']]
            geo_data['length'] = geo_data.apply(lambda row : row['path_geom'].length, axis = 1)
            geo_data['path'] = geo_data['path_geom'].apply(linestring_to_coords)
            geo_data["color"] = geo_data["line-color"].apply(hex_to_rgb)

            

            geo_data = pd.DataFrame(geo_data[['time', 'event', 'path', 'distance', 'color', 'id', 'length']])
            print(geo_data)
            filtered_data = geo_data.to_dict(orient='records')
            
            # start_loc = geo_data.iloc[0]['path'][0]
            # view_state = pdk.ViewState(latitude=start_loc[1], longitude=start_loc[0], zoom=10)
            # view_state = pdk.ViewState(latitude=54.093409, longitude=-2.89479, zoom=6)



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

def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color


def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16)*1 for i in (0, 2, 4)]

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
        
# UPDATES
def process_train_data(x):
    if len(x['td_messages'].values[0]) > 1:
        # print("TD Values found and loaded")
        df = pd.DataFrame(x['td_messages'].values[0])
        # print("sorting td df")
        df = geo_sort_tds_berths(df, berths_gdf_folium)
        if len(df) > 1:
            change = 99
            attempt = 0
            while change != 0 and attempt < 10:
                attempt+=1
                no_events = len(df)
                df = fill_in_gaps(df)
                change = no_events - len(df)
                print(f"Attempt: {attempt}, Change: {change} / {no_events}")
        df['path_geom'], df['path_ids'] = find_shortest_path(df, G, edges, nodes)
        elements = []
        df['path_ids'].apply(lambda x: [elements.append(a) for a in x] if (type(x) is list) else elements.append(x))
        # print("Returning train data")
    return df

#############################################################################################
def cut_line_at_point(line, pt):
    # First coords of line 
    if line.geom_type == 'LineString':
        coords = line.coords[:] 
    else:
        coords = [geo.coords[:] for geo in line.geoms]
        coords = [item for sublist in coords for item in sublist]
    # Add the coords from the points
    coords += pt.coords
    # Calculate the distance along the line for each point
    dists = [line.project(Point(p)) for p in coords]
    # sort the coordinates
    coords = [p for (d, p) in sorted(zip(dists, coords))]
    break_pt = find_index_of_point_w_min_distance(coords, pt.coords[:][0])

#     break_pt = coords.index(pt.coords[:][0])
    if break_pt == 0:
        # it is the first point on the line, "line_before" is meaningless
        line_before = None
    else:
        line_before = LineString(coords[:break_pt+1])
    if break_pt == len(coords)-1:
        # it is the last point on the line, "line_after" is meaningless

        line_after = None
    else:
        line_after = LineString(coords[break_pt:])
    return(line_before, line_after)

def distance_btw_two_points_on_a_line(line, pt1, pt2, bit="second"):

    if line.geom_type == 'LineString':
        coords = line.coords[:] 
    else:
        coords = [geo.coords[:] for geo in line.geoms]
        coords = [item for sublist in coords for item in sublist]


    # Add the coords from the points
    coords += pt1.coords
    coords += pt2.coords
    # Calculate the distance along the line for each point
    dists = [line.project(Point(p)) for p in coords]
    # sort the coordinates
    coords = [p for (d, p) in sorted(zip(dists, coords))]
    # get their orders
    first_pt = coords.index(pt1.coords[:][0])
    second_pt = coords.index(pt2.coords[:][0])
    if first_pt > second_pt :
        pt1, pt2 = pt2, pt1

    
    first_line_part = cut_line_at_point(line, pt1)[1]

    second_line_part = cut_line_at_point(first_line_part, pt2)[0]
    # distance = second_line_part.length
    if bit == 'first':
        return first_line_part
    elif bit == 'second':
        return second_line_part
    else:
        return second_line_part   

def find_index_of_point_w_min_distance(list_of_coords, coord):
    temp = [Point(c).distance(Point(coord)) for c in list_of_coords]
    return(temp.index(min(temp)) )


def find_shortest_path(df, graph, edges, nodes):
    geometries = []
    segments = []
    for _, row in df.iterrows():
        # print(_)
        from_point = row['from_coordinates']
        to_point = row['to_coordinates']
        from_edge_id = row['from_assetid']
        # print(from_edge_id, type(from_edge_id))
        to_edge_id = row['to_assetid']
        from_line = edges.loc[edges['ASSETID'] == from_edge_id]['geometry'].values[0]

        if from_edge_id == to_edge_id:
            td_step_path = distance_btw_two_points_on_a_line(from_line, from_point, to_point, "second")
            td_step_edges = from_edge_id
        else:
            to_line = edges.loc[edges['ASSETID'] == to_edge_id]['geometry'].values[0]
            td_step_edges = [from_edge_id]
            td_step_path, td_step_edges_route = route_between_lines(from_line, to_line, from_point, to_point, graph, edges, nodes)
            td_step_edges.extend(td_step_edges_route)
            td_step_edges.append(to_edge_id)

        geometries.append(td_step_path)
        segments.append(td_step_edges)

    return geometries, segments


def route_between_lines(from_line, to_line, from_point, to_point, graph, eds, ns):
    path = []
    path_parts = []
    path.append(get_start_end_line(from_line, from_point, to_point, "start"))


    start_node = from_line.boundary.geoms[1]
    end_node = to_line.boundary.geoms[0]
    shortest = ox.shortest_path(graph, ns.loc[ns['geometry'] == start_node].nodeID.values[0], ns.loc[ns['geometry'] == end_node].nodeID.values[0], weight='SHAPE_Leng')

    if shortest:
        edge_lines = []
        for i in range(len(shortest)-1):
            u = shortest[i]
            v = shortest[i+1]

            data = graph.get_edge_data(u, v)
            edge_lines.append(data[list(data.keys())[0]]['geometry'])
            path_parts.append(list(data.keys())[0])

        path.extend(edge_lines)

    path.append(get_start_end_line(to_line, to_point, from_point, "end"))

    multi_line = geometry.MultiLineString(path)
    merge_line = ops.linemerge(multi_line)

    return merge_line, path_parts




# def route_between_lines(from_line, to_line, from_point, to_point, graph, eds, ns):
#     path = []
#     path_parts = []
#     path.append(get_start_end_line(from_line, from_point, to_point, "start"))
#     path.append(get_start_end_line(to_line, to_point, from_point, "end"))

#     multi_line = geometry.MultiLineString(path)
#     merge_line = ops.linemerge(multi_line)
#     if merge_line.geom_type == 'MultiLineString':
#         start_node = from_line.boundary.geoms[1]
#         end_node = to_line.boundary.geoms[0]
#         shortest = ox.shortest_path(graph, ns.loc[ns['geometry'] == start_node].nodeID.values[0], ns.loc[ns['geometry'] == end_node].nodeID.values[0], weight='SHAPE_Leng')

#         for t in shortest:
#             print(ns.loc[ns['geometry'] == t])

#         if shortest:
#             edge_lines = []
#             for i in range(len(shortest)-1):
#                 u = shortest[i]
#                 v = shortest[i+1]

#                 data = graph.get_edge_data(u, v)
#                 edge_lines.append(data[list(data.keys())[0]]['geometry'])
#                 path_parts.append(list(data.keys())[0])
#             # Combine lines into a single LineString
#             # combined = geometry.MultiLineString(edge_lines)
#             # combined = ops.linemerge(combined)

#             # print(combined)
#             path.extend(edge_lines)
#             # path.append(combined)
#             multi_line = geometry.MultiLineString(path)
#             merge_line = ops.linemerge(multi_line)
#             # print("line merged")
#             return merge_line, path_parts
#     return multi_line, path_parts

def get_start_end_line(line, point1, point2, which_end):
    if which_end == 'start':
        if line.boundary.geoms[0].distance(point2) > line.boundary.geoms[1].distance(point2):
            _path = distance_btw_two_points_on_a_line(line, point1, line.boundary.geoms[1], "second")
        else:
            _path = distance_btw_two_points_on_a_line(line, line.boundary.geoms[0], point1, "second")
        return _path
    elif which_end == 'end':
        if line.boundary.geoms[0].distance(point2) < line.boundary.geoms[1].distance(point2):
            _path = distance_btw_two_points_on_a_line(line, line.boundary.geoms[0], point1, "second")
        else:
            _path = distance_btw_two_points_on_a_line(line, point1, line.boundary.geoms[1], "second")
        return _path


def fill_in_gaps(df):
    # Create a new list to hold the rows
    rows = []
    # Iterate over rows
    rows.append(df.iloc[0].values)
    for i in range(df.shape[0] - 1):
        if i >= 1:
            # print(f"Route: {df.iloc[i]['from']} /  {df.iloc[i]['to']}- Distance: {df.iloc[i]['from_coordinates'].distance(df.iloc[i]['to_coordinates'])}")
            if df.iloc[i]['from_coordinates'].distance(df.iloc[i]['to_coordinates']) < 0.1:    #UPDATE
                # Append the current row
                print("Location", i, "-", df.iloc[i]['from'], df.iloc[i]['to'], df.iloc[i]['from_coordinates'].distance(df.iloc[i]['to_coordinates']))
                rows.append(df.iloc[i].values)

                # Check if there's a gap
                if df.iloc[i]['to'] != df.iloc[i + 1]['from']:
                    # Create a new row and append it
                    new_row = [
                                int(((((int(df.iloc[i + 1]['time']) - int(df.iloc[i]['time'])) / 2)) + int(df.iloc[i]['time']))),
                                df.iloc[i]['to'] + "_" + df.iloc[i + 1]['from'],
                                df.iloc[i]['to'],
                                df.iloc[i + 1]['from'],
                                df.iloc[i]['to_coordinates'],
                                df.iloc[i + 1]['from_coordinates'],
                                df.iloc[i]['to_assetid'],
                                df.iloc[i + 1]['from_assetid'],
                                None,
                                '#00FF00'		
                                ]
                    rows.append(new_row)
            else:
                print(f"ERROR - Route: {df.iloc[i]['from']} /  {df.iloc[i]['to']}- Distance: {df.iloc[i]['from_coordinates'].distance(df.iloc[i]['to_coordinates'])}")


    # Append the last row
    # if df.iloc[-1]['from_coordinates'].distance(df.iloc[-1]['to_coordinates']) < 1.0:
    rows.append(df.iloc[-1].values)

    # Create a new DataFrame
    df_new = gpd.GeoDataFrame(rows, columns=df.columns)

    return df_new

def geo_sort_tds_berths(df, berths):
    events_with_geom = df.merge(berths, left_on='from', right_on='berth_id', suffixes=('', '_from'))
    events_with_geom = events_with_geom.merge(berths, left_on='to', right_on='berth_id', suffixes=('', '_to'))

    events = pd.DataFrame()
    events['time'] = events_with_geom['time']
    events['event'] = events_with_geom['step']
    events['from'] = events_with_geom['from']
    events['to'] = events_with_geom['to']
    events['from_coordinates'] = events_with_geom['geometry']
    events['to_coordinates'] = events_with_geom['geometry_to']
    events['from_assetid'] = events_with_geom['ASSETID']
    events['to_assetid'] = events_with_geom['ASSETID_to']
    events['path_geom'] = None
    events['line-color'] = '#FFFFFF'
    return events
#############################################################################################


if __name__ == '__main__':
    app.run_server(debug=True)

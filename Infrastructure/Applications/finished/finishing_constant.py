import os
import pickle
import pymongo
import pandas as pd
os.environ['USE_PYGEOS'] = '0'
from shapely.geometry import Point, LineString, MultiLineString
from shapely import geometry, ops
import osmnx as ox
import geopandas as gpd

from multiprocessing import Pool
import multiprocessing as mp
import numpy as np

from datetime import datetime

MONGO_URL = ''
MONGO_URL_FE = ''
MONGO_DB = ''
ARCHIVE_TABLE_NAME='archive'
HISTORY_TABLE_NAME='history'
UTIL_TABLE_NAME='utilisation'

def mongo(url, db, table):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db = mongo_client[db]
    mongo_tbl = mongo_db[table]
    return mongo_tbl

archive_table = mongo(MONGO_URL, MONGO_DB, ARCHIVE_TABLE_NAME)
history_table = mongo(MONGO_URL_FE, MONGO_DB, HISTORY_TABLE_NAME)
util_table = mongo(MONGO_URL_FE, MONGO_DB, UTIL_TABLE_NAME)

def get_archive_trains(mongo_archive):
    resp = list(mongo_archive.find())
    return resp

def put_history_trains(mongo, train_history):
    try:
        mongo.insert_one(train_history)
    except Exception as e: 
        print("Exception - Error calling mongodb - put train hisotry")
        print(f"Mongo Exception - Function: put_history_trains - ID: {train_history['_id']} - {e}")

def put_utilisation_record(mongo, util_record):
    try:
        mongo.insert_one(util_record)
    except Exception as e: 
        print("Exception - Error calling mongodb - put train hisotry")
        print(f"Mongo Exception - Function: put_history_trains - ID: {util_record['_id']} - {e}")


def remove_archive_record(mongo, key):
    key = {'_id': key}
    try:
        response = mongo.delete_one(key)
    except Exception as e: 
        print("Exception - Error calling mongodb - remove item in activated table")
        print(f"Mongo Exception - Function: remove_activated_record - ID: {key} - {e}")
    if response.acknowledged:
        print(f"Successfully processed {key}")

def process_train_data(x):
    data_load_path = 'utilisation_data_v1.pkl'
    with open(data_load_path, 'rb') as file:
        # Call load method to deserialze
        utilisation_data = pickle.load(file)
    G, nodes, edges, berths_gdf_folium = utilisation_data


    # print("TD Values found and loaded")
    df = pd.DataFrame(x['td_messages'])
    # print("sorting td df")
    df = geo_sort_tds_berths(df, berths_gdf_folium)
    if len(df) > 1:
        change = 99
        attempt = 0
        while change != 0 and attempt < 5:
            attempt+=1
            no_events = len(df)
            df = fill_in_gaps(df)
            change = no_events - len(df)
    df['path_geom'], df['path_ids'] = find_shortest_path(df, G, edges, nodes)
    elements = []
    df['path_ids'].apply(lambda x: [elements.append(a) for a in x] if (type(x) is list) else elements.append(x))
    # print("Returning train data")
    # unique_elements = list(set(elements))
    return df, elements


def geo_sort_tds_berths(df, berths):
    events_with_geom = df.merge(berths, left_on='from', right_on='berth_id', suffixes=('', '_from'))
    events_with_geom = events_with_geom.merge(berths, left_on='to', right_on='berth_id', suffixes=('', '_to'))

    events = pd.DataFrame()
    events['time'] = events_with_geom['time']
    events['event'] = events_with_geom['step'] if 'step' in events_with_geom else events_with_geom['from'] + "_" + events_with_geom['to']
    events['from'] = events_with_geom['from']
    events['to'] = events_with_geom['to']
    events['from_coordinates'] = events_with_geom['geometry']
    events['to_coordinates'] = events_with_geom['geometry_to']
    events['from_assetid'] = events_with_geom['ASSETID']
    events['to_assetid'] = events_with_geom['ASSETID_to']
    events['path_geom'] = None
    events['line-color'] = '#FFFFFF'
    return events

def put_util_tt(util_input, mongo):
    try:
        mongo.insert_one(util_input)
    except Exception as e: 
        print("Exception - Error calling mongodb - put item in running table")
        print(f"Mongo Exception - Function: put_finished_tt - ID: {util_input['_id']} - {e}")

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
    path.append(get_start_end_line(to_line, to_point, from_point, "end"))

    multi_line = geometry.MultiLineString(path)
    merge_line = ops.linemerge(multi_line)
    if merge_line.geom_type == 'MultiLineString':
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

            # Combine lines into a single LineString
            combined = LineString([p for line in edge_lines for p in line.coords])
            # print(combined)
            path.append(combined)
            multi_line = geometry.MultiLineString(path)
            merge_line = ops.linemerge(multi_line)
            # print("line merged")
            return merge_line, path_parts
    return multi_line, path_parts

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
    for i in range(df.shape[0] - 1):

        if df.iloc[i]['from_coordinates'].distance(df.iloc[i]['to_coordinates']) < 0.5:
            # Append the current row
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

    # Append the last row
    if df.iloc[-1]['from_coordinates'].distance(df.iloc[-1]['to_coordinates']) < 1.0:
        rows.append(df.iloc[-1].values)

    # Create a new DataFrame
    df_new = gpd.GeoDataFrame(rows, columns=df.columns)

    return df_new

def process_finished_trains(trains):
    start = datetime.now()
    print(f"Starting - chunk - {mp.current_process()._identity[0]} - No of Trains {len(trains)}")
    for train in trains:
        if 'td_messages' in train:
            if len(train['td_messages']) > 3:
                print(f"{train['_id']} - Status: {train['status']} - Activation: {train['activated_by']} - TDs: {len(train['td_messages'])}")
                try:
                    geo_data, geo_elements = process_train_data(train)

                    if geo_data['path_geom'].isnull().values.any():
                        geo_data = geo_data[geo_data['path_geom'].notna()]

                    geo_data = gpd.GeoDataFrame(geo_data, geometry='path_geom')

                    train['geo_data'] = geo_data[['time', 'event', 'from', 'to', 'path_geom', 'line-color', 'path_ids']].to_json()

                    utilisation = {'_id': train['_id'],
                                    'elements': geo_elements,
                                    'running_date': int(train['running_date'])
                                    }
                    put_history_trains(history_table, train)
                    put_utilisation_record(util_table, utilisation)
                    remove_archive_record(archive_table, train['_id'])
                except Exception as e: 
                    print(f"Exception - Function: process_train_data - ID: {train['_id']} - Removing Train - {e}")
                    remove_archive_record(archive_table, train['_id'])
            else:
                print(f"{train['_id']} - has only one TD message, not processed and simply to remove")
                remove_archive_record(archive_table, train['_id'])

        else:
            print(f"{train['_id']} - has no td messages, not processed and simply to remove")
            remove_archive_record(archive_table, train['_id'])
    finish = datetime.now()
    print(f"Ending - {mp.current_process()._identity[0]} - No of Trains: {len(trains)} - Duration: {str(finish-start)} - Rate: {len(trains)/(finish-start).total_seconds()}")

if __name__ == '__main__':
    
    finished_trains = get_archive_trains(archive_table)

    # Specify the number of processes to use in the pool
    num_processes = mp.cpu_count()
    # Split the DataFrame into chunks of size 'chunk_size'
    chunk_size = 1*num_processes

    chunks = np.array_split(finished_trains, round(len(finished_trains)/chunk_size))
   
    # Create a pool of processes
    pool = Pool(processes=num_processes)

    # Apply the modification function to each chunk in parallel
    modified_chunks = pool.map(process_finished_trains, chunks)

    # Close the pool to free resources
    pool.close()
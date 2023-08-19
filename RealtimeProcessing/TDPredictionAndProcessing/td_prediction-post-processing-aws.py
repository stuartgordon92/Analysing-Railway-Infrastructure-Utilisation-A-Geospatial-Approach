import pandas as pd
import numpy as np
import datetime as dt
import pymongo
import pickle
import os.path
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json

import scipy.stats as stats
from scipy.signal import argrelextrema

from datetime import datetime, timedelta
import pytz

from multiprocessing import Pool
import multiprocessing as mp

import time
import os
os.environ['USE_PYGEOS'] = '0'
from shapely.geometry import Point, LineString, MultiLineString
from shapely import geometry, ops
import osmnx as ox
import geopandas as gpd
from shapely.ops import linemerge

timezone = 'Europe/London'

MONGO_URL=os.getenv'mongo_URL'
MONGO_URL_FE=os.getenv'mongo_URL_FE'
MONGO_DB=os.getenv'mongo_DB'
ARCHIVE_TABLE_NAME='history'
HIST_TABLE_NAME='history_v2'
UTIL_TABLE_NAME='utilisation_v2'
TRANS_TABLE_NAME='transition_berth'

print("Loaded packages...")

def mongo(url, db, table):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db = mongo_client[db]
    mongo_tbl = mongo_db[table]
    return mongo_tbl

archive_table = mongo(MONGO_URL_FE, MONGO_DB, ARCHIVE_TABLE_NAME)
util_table = mongo(MONGO_URL_FE, MONGO_DB, UTIL_TABLE_NAME)
hist_table = mongo(MONGO_URL_FE, MONGO_DB, HIST_TABLE_NAME)
trans_berth_table = mongo(MONGO_URL_FE, MONGO_DB, TRANS_TABLE_NAME)


def process_train_data(df, G, nodes, edges, berths_gdf_folium):
    # data_load_path = 'utilisation_data_v1.pkl'
    # with open(data_load_path, 'rb') as file:
    #     # Call load method to deserialze
    #     utilisation_data = pickle.load(file)
    # G, nodes, edges, berths_gdf_folium = utilisation_data

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
    df = fill_in_gaps(df)
    df['path_geom'], df['path_ids'] = find_shortest_path(df, G, edges, nodes)
    df['path_geom'] = df['path_geom'].apply(merge_multilinestrings)
    elements = []
    df['path_ids'].apply(lambda x: [elements.append(a) for a in x] if (type(x) is list) else elements.append(x))

    df = calculate_speed(df)

    return df, elements

def merge_multilinestrings(row):
    if isinstance(row, MultiLineString):
        return linemerge(row)
    else:
        return row

def calculate_speed(df):
    # calculate distance of each path_geom
    df['distance'] = df['path_geom'].apply(lambda x: x.length*1e5 if x else 0)

    # drop the last row since it does not have a next row to compare with
    df = df[:-1]

    # calculate speed in km/h = distance_km / duration in hours
    df['speed_ms'] = df['distance'] / df['duration_actual']

    # convert speed to mph = speed_kmh * 0.621371
    df['speed_mph'] = df['speed_ms'].copy() * 2.23694

    return df

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

    events['duration_actual'] = events_with_geom['time_in_berth_actual']
    events['duration_predict'] = events_with_geom['time_in_berth_predict']
    events['flag'] = events_with_geom['event']

    return events

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

def print_metrics_prediction(mae_start, mse_start, rmse_start, r2_start, mape_start):
    print("-"*50)
    print(f"Performance Metrics for Time in Berth Prediction")
    print(f"Mean Absolute Error for berth time: {mae_start:.4f} seconds")
    print(f"Mean Squared Error for berth time: {mse_start:.4f} (seconds)^2")
    print(f"Root Mean Squared Error for berth time: {rmse_start:.4f} seconds")
    print(f"R-Squared for berth time: {r2_start:.4f}")
    print(f"Mean Absolute Percentage Error for berth time: {mape_start:.4f}%")

def print_metrics_timings(mae_start, mae_end, mse_start, mse_end, rmse_start, rmse_end, r2_start, r2_end, mape_start, mape_end):
    print("-"*50)
    print(f"Performance Metrics for Start/End Timing Prediction")
    print(f"Mean Absolute Error for start times: {mae_start:.2f} seconds")
    print(f"Mean Absolute Error for end times: {mae_end:.2f} seconds\n")

    print(f"Mean Squared Error for start times: {mse_start:.2f} (seconds)^2")
    print(f"Mean Squared Error for end times: {mse_end:.2f} (seconds)^2\n")

    print(f"Root Mean Squared Error for start times: {rmse_start:.2f} seconds")
    print(f"Root Mean Squared Error for end times: {rmse_end:.2f} seconds\n")

    print(f"R-Squared for start times: {r2_start:.2f}")
    print(f"R-Squared for end times: {r2_end:.2f}\n")

    print(f"Mean Absolute Percentage Error for start times: {mape_start:.2f}%")
    print(f"Mean Absolute Percentage Error for end times: {mape_end:.2f}%")

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def assign_event(df, df_pred):
    # Merge the two dataframes on 'id_step' and 'headcode'
    merged_df = df.rename(columns={'descr':'headcode'}).merge(df_pred, on=['id_step', 'headcode'], how='inner')

    # Define a function that assigns 'Significantly Early', 'Significantly Delayed', or 'Normal'
    def assign_status(row):
        if row['time_in_berth'] < row['mean_time_in_berth_early_threshold']:
            return 'Significantly Early'
        elif row['time_in_berth'] > row['mean_time_in_berth_late_threshold']:
            return 'Significantly Delayed'
        else:
            return 'Normal'

    # Apply the function to every row
    merged_df['event'] = merged_df.apply(assign_status, axis=1)
    return merged_df

def compare_planned_to_actual(act, res, pred, train_data):

    act = assign_event(act, pred)
    df = pd.merge(act, res, on='id_step', how='right', suffixes=('_actual', '_predict'))
    df = df.loc[(df['time_in_berth_actual'] > 0) & (df['time_in_berth_actual'] < 3600)]

    df['pred_diff'] = (df['time_in_berth_actual'] - df['time_in_berth_predict'])
    df['time_in_berth_actual_norm'] = (df['time_in_berth_actual'] - df['time_in_berth_actual'].min()) / (df['time_in_berth_actual'].max() - df['time_in_berth_actual'].min())
    df['time_in_berth_predict_norm'] = (df['time_in_berth_predict'] - df['time_in_berth_predict'].min()) / (df['time_in_berth_predict'].max() - df['time_in_berth_predict'].min())

    df['start_diff'] = (df['start_time_actual'] - df['start_time_predict']).dt.total_seconds()
    df['end_diff'] = (df['end_time_actual'] - df['end_time_predict']).dt.total_seconds()

    df['timetable_variation'] = df['timetable_variation'].astype('int32')
    df['timetable_variation_scaled'] = 0.1 + 0.6*((df['timetable_variation'] - df['timetable_variation'].min()) / (df['timetable_variation'].max() - df['timetable_variation'].min()))

    df['difference_norm'] = df['time_in_berth_actual_norm'] - df['time_in_berth_predict_norm']
    df['difference'] = df['time_in_berth_actual'] - df['time_in_berth_predict']

    df_stat = {
        'no_pred_td': len(res)
    }
    events = df['event'].value_counts().to_dict()
    df_stats = df_stat | events


    return df, df_stats

def score_timings(df_in):
    results_dict = {
            'mae_predict': None,
            'mse_predict': None,
            'rmse_predict': None,
            'r2_predict': None,
            'mape_predict': None,
            'plan_mae_predict': None,
            'plan_mse_predict': None,
            'plan_rmse_predict': None,
            'plan_r2_predict': None,
            'plan_mape_predict': None,
        }

    if len(df_in) > 1:
        df = df_in.copy(deep=True)
        df['time_in_berth_predict'] = df['time_in_berth_predict'].astype('int64') / 10**9  # convert to seconds
        df['time_in_berth_actual'] = df['time_in_berth_actual'].astype('int64') / 10**9  # convert to seconds

        # timings
        mae_predict = mean_absolute_error(df['time_in_berth_predict'], df['time_in_berth_actual'])
        mse_predict = mean_squared_error(df['time_in_berth_predict'], df['time_in_berth_actual'])
        rmse_predict = np.sqrt(mean_squared_error(df['time_in_berth_predict'], df['time_in_berth_actual']))
        r2_predict = r2_score(df['time_in_berth_predict'], df['time_in_berth_actual'])
        mape_predict = mean_absolute_percentage_error(df['time_in_berth_actual'], df['time_in_berth_predict'])

        results_dict['mae_predict'] = mae_predict
        results_dict['mse_predict'] = mse_predict
        results_dict['rmse_predict'] = rmse_predict
        results_dict['r2_predict'] = r2_predict
        results_dict['mape_predict'] = mape_predict


        df = df.dropna(subset=['planned_timestamp_offset'])
        if len(df) > 1:
            df['start_time_predict'] = df['start_time_predict'].astype('int64') / 10**9  # convert to seconds
            df['planned_timestamp_offset'] = df['planned_timestamp_offset'].astype('int64') / 10**9  # convert to seconds

            # timings
            plan_mae_predict = mean_absolute_error(df['start_time_predict'], df['planned_timestamp_offset'])
            plan_mse_predict = mean_squared_error(df['start_time_predict'], df['planned_timestamp_offset'])
            plan_rmse_predict = np.sqrt(mean_squared_error(df['start_time_predict'], df['planned_timestamp_offset']))
            plan_r2_predict = r2_score(df['start_time_predict'], df['planned_timestamp_offset'])
            plan_mape_predict = mean_absolute_percentage_error(df['planned_timestamp_offset'], df['start_time_predict'])

            results_dict['plan_mae_predict'] = plan_mae_predict
            results_dict['plan_mse_predict'] = plan_mse_predict
            results_dict['plan_rmse_predict'] = plan_rmse_predict
            results_dict['plan_r2_predict'] = plan_r2_predict
            results_dict['plan_mape_predict'] = plan_mape_predict

    return results_dict

def process_raw_td(df):
    df['Timestamp'] = pd.to_datetime(df['datetime'], format='ISO8601', utc=True)
    df['Epoch'] = df['Timestamp'].astype('int64')//1e9
    df['Service'] = df['headcode'].str[0:2]

    df_clean = df.loc[(df['msg_type'] == 'CA') &
                    (df['time_in_berth'] >= 1) & 
                    (df['time_in_berth'] <= 1800) &
                    (df['headcode'].str.contains(r'[0-9][A-Z][0-9][0-9]')) &
                    (df['next_step_in_area'] != '')
                    ]
    df_clean.drop(['datetime', 'berth_from', 'id_berth_from', 'berth_to', 'id_berth_to', 'msg_type'], axis=1, inplace=True)
    df_clean.drop_duplicates(inplace=True)
    return df_clean

def create_predict_df(df_clean, confidence_level=0.95):
    print(f"{dt.datetime.now()} - Starting Processing")
    # Apply several functions to the 'time_in_berth' column
    df_predict = df_clean.groupby(['id_step', 'headcode']).agg({
        'next_step_in_area': pd.Series.mode,
        'time_in_berth': [np.mean, 'median', 'var', 'std', 'count', lambda x: x.tolist()]
    }).reset_index()
    print(f"{dt.datetime.now()} - Group By Complete")
    # Flatten column index
    df_predict.columns = ['_'.join(col).strip() for col in df_predict.columns.values]
    print(f"{dt.datetime.now()} - Flatten Coloumn Index")
    # Rename new columns
    df_predict.rename(columns={
        'next_step_in_area_<lambda_0>': 'next_step_in_area_values',
        'time_in_berth_mean': 'mean_time_in_berth', 
        'time_in_berth_median': 'median_time_in_berth', 
        'time_in_berth_var': 'variance_time_in_berth', 
        'time_in_berth_std': 'std_dev_time_in_berth', 
        'time_in_berth_count': 'samples_count', 
        'time_in_berth_<lambda_0>': 'samples_values',
        'id_step_': 'id_step', 
        'headcode_': 'headcode'
    }, inplace=True)
    print(f"{dt.datetime.now()} - Renamed Coloumns")
    # Extract single mode value
    non_array = []
    for index, value in df_predict['next_step_in_area_mode'].items():
        non_array.append(value) if isinstance(value, str) else non_array.append(np.nan)
    df_predict['next_step_in_area'] = non_array
    print(f"{dt.datetime.now()} - Next Step Complete")
    df_predict['distribution'] = df_predict.apply(lambda row: classify_distribution(row['samples_values']) if row['samples_count'] >= 8 else 'insufficient_data', axis=1)
    print(f"{dt.datetime.now()} - Distribution Calculated - {len(df_predict)}")
    # Compute the confidence intervals for mean and median
    df_predict[['mean_time_in_berth_early_threshold', 'mean_time_in_berth_late_threshold']] = df_predict.apply(lambda row: compute_confidence_interval(row['samples_values'], row['mean_time_in_berth'], row['distribution'], confidence_level), axis=1, result_type='expand')
    # df_predict[['median_time_in_berth_early_threshold', 'median_time_in_berth_late_threshold']] = df_predict['samples_values'].apply(lambda row: compute_bootstrap_CI(row, confidence_level), axis=1, result_type='expand')
    print(f"{dt.datetime.now()} - Confidence Intervals Calculated")
    return df_predict[['id_step', 'headcode', 'next_step_in_area_mode', 'mean_time_in_berth',
       'median_time_in_berth', 'variance_time_in_berth',
       'std_dev_time_in_berth', 'samples_count',
       'next_step_in_area', 'distribution',
       'mean_time_in_berth_early_threshold',
       'mean_time_in_berth_late_threshold']]

def compute_confidence_interval(values, mean, distribution, confidence_level):
    if distribution == 'normal':
        # Calculate standard deviation
        std_dev = np.std(values, ddof=1)

        # Calculate confidence interval using mean x 2 x standard deviation
        ci_lower = mean - 2 * std_dev
        ci_upper = mean + 2 * std_dev
    else:
        # Use percentile-based method for non-normal data
        ci_lower = np.percentile(values, (1 - confidence_level) / 2 * 100)
        ci_upper = np.percentile(values, (1 + confidence_level) / 2 * 100)

    return ci_lower, ci_upper

def classify_distribution(values):
    # Apply normal test
    try:
        k2, p = stats.normaltest(values)
    except ValueError:
        return 'insufficient_data'
    
    # If the p-value is larger than 0.05, it's normal distribution
    if p > 0.05:
        return 'normal'
    
    # Let's use some thresholds to classify other types of distribution
    skewness = stats.skew(values)
    if abs(skewness) < 0.5:
        return 'uniform'
    elif skewness < 0:
        return 'skewed_left'
    elif skewness > 0:
        return 'skewed_right'
    
    return 'unknown'

def count_local_maxima(values):
    # Transform the values to a Numpy array
    values = np.array(values)
    
    # Calculate local maxima
    local_maxima = argrelextrema(values, np.greater)
    
    # Return the count of local maxima
    return len(local_maxima[0])

def make_prediction_model(input_df):
    cleaned_data = process_raw_td(input_df)
    model_df = create_predict_df(cleaned_data)

    return model_df

def upload_trans_berth(mongo, from_berth_step, to_berth_step):
    try:
        mongo.insert_one({
            'from_berth_step': from_berth_step,
            'to_berth_step': to_berth_step,
            'source': 'td_prediction_post',
            'added': datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        })
    except Exception as e: 
        print("Exception - Error calling mongodb - upload_trans_berth")
        print(f"Mongo Exception - Function: upload_trans_berth - ID: {from_berth_step} - {to_berth_step} - {e}")
 
def find_new_transition_berth(current_id, id_list, transition_df):
    try:
        # Find the current_id in the list
        # print("trying to find new transtion")
        current_index = id_list.index(current_id)
        # Iterate over the list starting from the next id
        for id_ in id_list[current_index + 1:]:
            # If the first two characters or the rest of the string are different
            if id_[:2] != current_id[:2] or id_[2:] != current_id[2:]:

                # Check if the id_ is in transition_df
                if id_ not in transition_df['different_id_step'].values or id_ not in transition_df['id_step'].values:
                    # If not, add a row with id_, current_id and 'inferred' as the source - id_step	different_id_step
                    new_row = pd.DataFrame({'id_step': [current_id], 'different_id_step': [id_], 'source': ['inferred']})
                    transition_df = pd.concat([transition_df, new_row], ignore_index=True)
                    print(f"----|||----- Added new transition berth: id_step: {current_id}, different_id_step: {id_}")
                    upload_trans_berth(trans_berth_table, current_id, id_)
                # Return this id
                else:
                    new_row = pd.DataFrame({'id_step': [current_id], 'different_id_step': [id_], 'source': ['inferred']})
                    transition_df = pd.concat([transition_df, new_row], ignore_index=True)
                    print(f"----|||----- Added new transition berth: id_step: {current_id}, different_id_step: {id_}")
                    upload_trans_berth(trans_berth_table, current_id, id_)
                return id_, transition_df

    except ValueError:
        # print("value error")
        # current_id not in the list
        return None, None

    # If no id found with different first two characters and different rest of string
    return None, None

def find_steps(id_step, headcode, start_time, grouped_df, end_step, train_steps, transitions):
    result_data = []
    # print(id_step, headcode, start_time, end_step, train_steps)
    previous_values = []
    count=0
    while True:
        # Check if all values are the same
        print(f"--- {headcode} --- {count} / {len(train_steps)} - {round(count/len(train_steps),2)*100}% --- {id_step} - {end_step}")
        if count > 250:
            print(f"Prediction Failed ({headcode}) - Infinite Loop in Area: {id_step} - {prev_step}")
            state = f'Prediction Failed ({headcode}) - Infinite Loop in Area - {id_step} - {prev_step}'
            break

        # Find the corresponding entry in the dataframe
        row = grouped_df[(grouped_df['id_step'] == id_step) & (grouped_df['headcode'] == headcode)]
        # Check if we found an entry
        if len(row) == 0:
            # Update the list of previous values
            previous_values.append(id_step)
            # print("---||--", previous_values)
            if len(previous_values) > 3:
                previous_values.pop(0)

            # Check if all values are the same
            if len(previous_values) == 3 and previous_values[0] == previous_values[1] == previous_values[2]:
                print(f"Prediction Failed ({headcode}) - Infinite Loop: {id_step} - {end_step}")
                state = f'Prediction Failed ({headcode}) - Infinite Loop - {id_step}'
                break


            # print(f"End Reached 1: {id_step} - {end_step}")
            if id_step == end_step:
                print(f"Predict to Final Berth ({headcode}) (1): {id_step} - {end_step}")
                state = f'Predict to Final Berth ({headcode}) - {id_step}'
                break
            elif id_step[:2] == end_step[:2]:
                print(f"Predict to Final Area ({headcode}) (1): {id_step} - {end_step}")
                state = f'Predict to Final Area ({headcode}) - {id_step} - {end_step}'
                break
            else:
                # print(f"End Not Reached 1: {id_step}, try find new transition")
                new_id, transitions_update = find_new_transition_berth(id_step, train_steps, transitions)
                if new_id:
                    # print(f"End Not Reached 1: {id_step}, Found new transition berth {new_id}")
                    transitions = transitions_update
                    transition = transitions.loc[(transitions['id_step'] == id_step)]
                    row = grouped_df[(grouped_df['id_step'] == transition['different_id_step'].values[0]) & (grouped_df['headcode'] == headcode)]
                else:
                    # print(f"End Not Reached 1: {id_step}, try prev_step: {prev_step}")
                    new_id = None
                    new_id, transitions_update = find_new_transition_berth(prev_step, train_steps, transitions)
                    if new_id:
                        transitions = transitions_update
                        transition = transitions.loc[(transitions['id_step'] == prev_step)]
                        row = grouped_df[(grouped_df['id_step'] == transition['different_id_step'].values[0]) & (grouped_df['headcode'] == headcode)]
                    else:
                        # print(f"End Not Reached 1-1: {id_step}, try prev_step: {prev_step}")
                        transition = transitions.loc[(transitions['different_id_step'] == id_step)]
                        if len(transition) >= 1:
                            row = grouped_df[(grouped_df['id_step'] == transition['id_step'].values[0]) & (grouped_df['headcode'] == headcode)]
                        else:
                            # print(f"End Not Reached 1-2: {id_step}, try prev_step: {prev_step}")
                            transition = transitions.loc[(transitions['id_step'] == id_step)]
                            if len(transition) >= 1:
                                row = grouped_df[(grouped_df['id_step'] == transition['different_id_step'].values[0]) & (grouped_df['headcode'] == headcode)]
                            else:
                                print(f"Prediction Failed ({headcode}) (1-3): {id_step} - {end_step}")
                                state = f'Prediction Failed ({headcode}) - {id_step}'
                                break  

            if len(row) == 1:
                pass
                # print(f"Transition from : {id_step} - {row['id_step'].values[0]}")
            elif len(row) == 0:
                # print(f"--- Checking service code")
                row = grouped_df[(grouped_df['id_step'] == transition['id_step'].values[0]) & (grouped_df['headcode'].str[:2] == headcode[:2])]
                if len(row) == 0:
                    if id_step == end_step:
                        print(f"Predict to Final Berth ({headcode}) (3): {id_step} - {end_step}")
                        state = f'Predict to Final Berth ({headcode}) - {id_step}'
                        break
                    elif id_step[:2] == end_step[:2]:
                        print(f"LPredict to Final Area ({headcode}) (3): {id_step} - {end_step}")
                        state = f'Predict to Final Area ({headcode}) - {id_step} - {end_step}'
                        break
                    else:
                        print(f"Prediction Failed ({headcode}) (3): {id_step} - {end_step}")
                        state = f'Prediction Failed ({headcode}) - {id_step}'
                        break
                else:
                    pass
                    # print(f"Transition from : {id_step} - {row['id_step'].values[0]}")
            else:
                # print(f"Error, dropping out - {id_step} - {row['id_step'].values}")

                if id_step == end_step:
                    print(f"Predict to Final Berth ({headcode}) (2): {id_step} - {end_step}")
                    state = f'Predict to Final Berth ({headcode}) - {id_step}'
                    break
                elif id_step[:2] == end_step[:2]:
                    print(f"Predict to Final Area ({headcode}) (2): {id_step} - {end_step}")
                    state = f'Predict to Final Area ({headcode}) - {id_step} - {end_step}'
                    break
                else:
                    print(f"Prediction Failed ({headcode}) (2): {id_step} - {end_step}")
                    state = f'Prediction Failed ({headcode}) - {id_step}'
                    break

        if id_step == end_step:
            print(f"Predict to Final Berth ({headcode}) (4): {id_step} - {end_step}")
            state = f'Predict to Final Berth ({headcode}) - {id_step}'
            break

        # Get the next step and time
        prev_step = id_step
        next_step = row['next_step_in_area'].values[0]
        step_time = row['mean_time_in_berth'].values[0]

        # Append to result list
        result_data.append([id_step, headcode, next_step, start_time, start_time + step_time])

        # Update start time
        start_time = start_time + step_time
        
        # print("----", id_step, next_step)

        # Update id_step for the next iteration
        id_step = next_step
        state = 'in progress'
        count+=1
        

    # Convert list to DataFrame
    result_df = pd.DataFrame(result_data, columns=['id_step', 'headcode', 'next_step', 'start_time', 'end_time'])
    result_df['time_in_berth'] = result_df['end_time'] - result_df['start_time']
    result_df['start_time'] = result_df['start_time'].apply(lambda x: dt.datetime.fromtimestamp(x))
    result_df['end_time'] = result_df['end_time'].apply(lambda x: dt.datetime.fromtimestamp(x))
    result_df = result_df.rename(columns={'next_step': 'next_step_in_area'})
    return result_df, state

def calculate_time_diff_and_next_step(df):
    # Convert epoch milliseconds to seconds
    df = df.loc[df['msg_type'] == 'CA']
    df['time'] = df['time'].copy().astype('int64')/ 1000
    df = df.reset_index(drop=True)
    # Initialize new columns
    df['time_in_berth'] = np.nan
    df['next_step_in_area'] = np.nan

    # Go through each row (except the last one)
    for i in range(len(df) - 1):
        # Get the current row's 'to' and 'descr'
        current_to = df.loc[i, 'to']
        current_descr = df.loc[i, 'descr']

        # Find the next row that meets the condition
        next_row = df[(df['from'] == current_to) & (df['descr'] == current_descr) & (df.index > i)]

        # Check if a matching row is found
        if not next_row.empty:
            # Get the index of the next matching row
            next_index = next_row.index[0]

            # Calculate time difference
            df.loc[i, 'time_in_berth'] = df.loc[next_index, 'time'] - df.loc[i, 'time']

            # Get next step
            df.loc[i, 'next_step_in_area'] = df.loc[next_index, 'step']

    df.dropna(inplace=True)
    
    df['start_time'] = df['time'].apply(lambda x: dt.datetime.fromtimestamp(x))
    df['end_time'] = df['time'] + df['time_in_berth']
    df['end_time'] = df['end_time'].apply(lambda x: dt.datetime.fromtimestamp(x))

    df = df.rename(columns={
        'time':'Timestamp',
        'step': 'id_step'
        })
    return df

def planned_predict(x, prediction_df, transition_df):
    for td in x['td_messages']:
        if td['msg_type'] == 'CA':
            start_step = td['step']
            headcode = td['descr']
            start_time = int(int(td['time'])/1e3)
            break

    for td in reversed(x['td_messages']):
        if td['msg_type'] == 'CA':
            end_step = td['step']
            # end_time = int(int(td['time'])/1e3)
            break
    
    ca = []
    for td in x['td_messages']:
        if td['msg_type'] == 'CA':
            ca.append(td['step'])

    actual_df = pd.DataFrame(x['td_messages'])
    actual_df = calculate_time_diff_and_next_step(actual_df)
    actual_df['train_id'] = x['id']

    start_time = int(int(x['planned_timetable'][0]['planned_timestamp'])/1e3)
    start_time_dst_test = pytz.timezone(timezone).localize(datetime.fromtimestamp(start_time))
    if is_dst(start_time_dst_test):
        start_time = start_time - 3600

    result_df, state = find_steps(start_step, headcode, start_time, prediction_df, end_step, ca, transition_df)
    result_df['train_id'] = x['id']
    
    return actual_df, result_df, state, len(ca)

def mongo(url, db, table):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db = mongo_client[db]
    mongo_tbl = mongo_db[table]
    return mongo_tbl

def get_train_data(mongo, id=None):
    if id:
        query = {'id': id}
        resp = list(mongo.find(query))
        if len(resp) == 1:
            resp = resp[0]
    else:
        resp = mongo.find_one()

    if len(resp) > 0:
        if 'td_messages' in resp:
            if len(resp['td_messages']) > 2:
                print(f"Train {resp['signalling_id']}({resp['atoc_code']}) - {resp['CIF_train_uid']} running on {resp['running_date']} - from {resp['origin_location']} to {resp['destination_location']} ({len(resp['td_messages'])}) - {resp['status']}")
            else:
                print(f"Train {resp['signalling_id']}({resp['atoc_code']}) found but there is an issue with the data: {resp['status']}")
                return None
        else:
            print(f"Train {resp['signalling_id']}({resp['atoc_code']}) found but there is an issue with the data: {resp['status']}")
            return None
    else:
        print(f"Train {id} does not exist in database")
        return None

    return resp

def predict_and_compare_service(id, df_predict, transitions, berth_to_loc):
    
    train_data = get_train_data(archive_table, id=id)
    if train_data is not None:
        actual, result, state, no_events = planned_predict(train_data, df_predict, transitions)
        if len(result) >= 2 and len(actual) >= 2:
            compare_df, compare_stats = compare_planned_to_actual(actual, result, df_predict, train_data)
            compare_df = compare_to_tt(compare_df, train_data, berth_to_loc)
            results = score_timings(compare_df)
            results['train_id'] = id
            results['trains_title'] = f"Train {train_data['signalling_id']}({train_data['atoc_code']}) - {train_data['CIF_train_uid']} running on {train_data['running_date']} - from {train_data['origin_location']} to {train_data['destination_location']} ({len(train_data['td_messages'])}) - {train_data['status']}"
            results['prediction_result'] = state
            results['actual_no_actual_ca'] = no_events
            results['actual_no_pred_ca'] = compare_stats['no_pred_td']
            results['actual_no_pred_matches'] = len(compare_df)
            results['actual_prediction_matches_accuracy'] = results['actual_no_pred_matches'] / results['actual_no_pred_ca']
            # results['actual_prediction_accuracy'] = results['actual_no_pred_matches'] / results['actual_no_actual_ca']

            results['actual_prediction_precision'] = results['actual_no_pred_matches'] / (results['actual_no_actual_ca'])
            results['actual_prediction_recall'] = results['actual_no_pred_matches'] / results['actual_no_pred_ca']
            try:
                results['actual_prediction_F1'] = 2* (results['actual_prediction_precision'] * results['actual_prediction_recall']) / (results['actual_prediction_precision'] + results['actual_prediction_recall'])
            except:
                results['actual_prediction_F1'] = None


            results['plan_no_actual_locs'] = len(train_data['actual_timetable'])
            results['plan_no_plan_loc_matches'] = len(compare_df['planned_timestamp'].dropna())
            results['plan_prediction_precision'] = results['plan_no_plan_loc_matches'] / results['plan_no_actual_locs']

            return compare_df, (results | compare_stats), train_data
        else:
            return None, None, train_data
    else:
        return None, None, None
    
def compare_to_tt(df, t_data, berth_to_loc):
    plan_comp = df[['Timestamp',
        'id_step', 'timetable_variation', 'time_in_berth_actual','time_in_berth_predict',
        'start_time_actual', 'end_time_actual', 'start_time_predict', 'end_time_predict',
        'next_step_in_area', 'event', 'pred_diff'
        ]].merge(berth_to_loc.rename(columns={'event':'event_type'}), on='id_step',how='left').dropna().merge(
                    pd.DataFrame(t_data['actual_timetable'])[['loc_stanox','event_type','planned_timestamp']].astype({'planned_timestamp':'datetime64[ms]'}),
                    on=['loc_stanox','event_type'],
                    how='left').dropna()
    plan_comp['planned_timestamp_offset'] = plan_comp['planned_timestamp'] + plan_comp['offset'].astype("timedelta64[s]")
    plan_comp['plan_predict_diff'] = ((plan_comp['planned_timestamp'] - plan_comp['start_time_predict']).dt.total_seconds())
    plan_comp['plan_predict_diff_offset'] = ((plan_comp['planned_timestamp'] - plan_comp['start_time_predict']).dt.total_seconds() + plan_comp['offset'])
    plan_comp = plan_comp[['id_step','loc_stanox', 'event_type', 'offset', 'Name',
       'planned_timestamp', 'plan_predict_diff', 'plan_predict_diff_offset', 'planned_timestamp_offset']]
    df = df.merge(plan_comp, on='id_step', how='left')
    return df

def get_predict_model(td_filename):
    if os.path.exists(f'prediction_model_{td_filename}.pkl'):
        # print('Found model')
        with open(f'prediction_model_{td_filename}.pkl', 'rb') as file:
            # Call load method to deserialze
            df_predict = pickle.load(file)
    else:
        print('No model found - creating new')
        df = pd.read_csv(td_filename)
        df_predict = make_prediction_model(df)
        df_predict.rename(columns={'id_step_':'id_step',
                            'headcode_':'headcode'
        }, inplace=True)
        with open(f'prediction_model_{td_filename}.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(df_predict, file)
    df_predict = df_predict.dropna(subset=['next_step_in_area','mean_time_in_berth'])
    return df_predict

def get_berth_to_loc_model(berth_to_loc_filename):
    if os.path.exists(f'{berth_to_loc_filename}.pkl'):
        # print('Found data')
        with open(f'{berth_to_loc_filename}.pkl', 'rb') as file:
            # Call load method to deserialze
            berth_to_loc = pickle.load(file)
    else:
        print('No model found - creating new')
        LOC_PATH = r'C:\Users\stuar\OneDrive - Newcastle University\Year 1\Thesis\Geospatial Model\raw\raw\TiplocPublicExport_2022-12-24_10-37 (1).json'
        with open(LOC_PATH, 'r') as fp:
            loc_data = json.load(fp)
        locs = pd.DataFrame.from_dict(loc_data["Tiplocs"])

        SMART_PATH = r'C:\Users\stuar\OneDrive - Newcastle University\Year 1\Thesis\Geospatial Model\raw\td_list\SMARTExtract.json'
        with open(SMART_PATH, 'r') as fp:
            smart_data = json.load(fp)

        smart = pd.DataFrame.from_dict(smart_data["BERTHDATA"])
        smart['id_step'] = smart['TD'] + "_" + smart['FROMBERTH'] + "_" + smart['TOBERTH']
        smart['event'] = smart['EVENT'].apply(lambda x: 'ARRIVAL' if x == 'A' or x == 'C' else 'DEPARTURE')
        smart = smart[['id_step', 'STANOX', 'event', 'BERTHOFFSET']].rename(columns={'STANOX': 'loc_stanox', 'BERTHOFFSET': 'offset'})
        smart['offset'] = smart['offset'].str.replace('+', '').astype(int)
        berth_to_loc = smart.astype({'loc_stanox':'int64'}).fillna(0).merge(locs[['Name', 'Stanox']].rename(columns={'Stanox':'loc_stanox'}).dropna().astype({'loc_stanox':'int64'}), on='loc_stanox', how='left')

        with open(f'{berth_to_loc_filename}.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(berth_to_loc, file)
    return berth_to_loc

def is_dst(when):
    '''Given the name of Timezone will attempt determine if that timezone is in Daylight Saving Time now (DST)'''
    return when.dst() != timedelta(0)

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

def remove_archive_record_id(mongo, key):
    key = {'id': key}
    try:
        response = mongo.delete_one(key)
    except Exception as e: 
        print("Exception - Error calling mongodb - remove item in activated table")
        print(f"Mongo Exception - Function: remove_activated_record - ID: {key} - {e}")
    if response.acknowledged:
        print(f"Successfully processed {key}")

def remove_archive_record(mongo, key):
    key = {'_id': key}
    try:
        response = mongo.delete_one(key)
    except Exception as e: 
        print("Exception - Error calling mongodb - remove item in activated table")
        print(f"Mongo Exception - Function: remove_activated_record - ID: {key} - {e}")
    if response.acknowledged:
        print(f"Successfully processed {key}")

def get_models():
    td_filename = 'full_20230522.csv'
    df_predict =  get_predict_model(td_filename)

    berth_to_loc_filename = 'berth_to_loc_20230723'
    berth_to_loc =  get_berth_to_loc_model(berth_to_loc_filename)

    transitions = pd.read_csv('transitons2.csv')
    return df_predict, berth_to_loc, transitions

def process_ids(ids):
    start = datetime.now()
    print(f"Starting - chunk - {mp.current_process()._identity[0]} - No of Trains {len(ids)}")
    df_predict, berth_to_loc, transitions = get_models()

    data_load_path = 'utilisation_data_v1.pkl'
    with open(data_load_path, 'rb') as file:
        # Call load method to deserialze
        utilisation_data = pickle.load(file)
    G, nodes, edges, berths_gdf_folium = utilisation_data


    for id in ids:
        print(f'{datetime.now().strftime("%H:%M:%S %d/%m/%Y")} - {id}')
        compare_df, results, train = predict_and_compare_service(id, df_predict.loc[df_predict['samples_count'] > 1], transitions, berth_to_loc)
        if compare_df is not None:
            try:
                geo_data, geo_elements = process_train_data(compare_df.rename(columns={'Timestamp':'time'}), G, nodes, edges, berths_gdf_folium)

                if geo_data['path_geom'].isnull().values.any():
                    geo_data = geo_data[geo_data['path_geom'].notna()]

                geo_data = gpd.GeoDataFrame(geo_data, geometry='path_geom')

                train['geo_data_prediction'] = geo_data[['time', 'event', 'from', 'to',
                                            'from_assetid', 'to_assetid', 'path_geom', 'line-color',
                                            'duration_actual', 'duration_predict', 'flag', 'path_ids', 'distance',
                                            'speed_ms', 'speed_mph']].to_json()
                train['prediction_results'] = results

                utilisation = {'_id': train['_id'],
                                'elements': geo_elements,
                                'running_date': int(train['running_date'])
                                }
                put_history_trains(hist_table, train)
                put_utilisation_record(util_table, utilisation)
                remove_archive_record(archive_table, train['_id'])
            except Exception as e: 
                print(f"Exception - Function: process_train_data - ID: {train['_id']} - Removing Train - {e}")
                remove_archive_record(archive_table, train['_id'])
        else:
            print(f"{id} - has only one TD message, not processed and simply to remove")
            if train:
                remove_archive_record(archive_table, train['_id'])
            else:
                remove_archive_record_id(archive_table, id)
    finish = datetime.now()
    print(f"Ending - {mp.current_process()._identity[0]} - No of Trains: {len(ids)} - Duration: {str(finish-start)} - Rate: {len(ids)/(finish-start).total_seconds()}")

def get_archive_trains(mongo_archive):
    resp = list(mongo_archive.find({},{"_id": 0, "id": 1}).limit(100000))
    trains = []
    for r in resp:
        trains.append(r['id'])
    return trains

if __name__ == '__main__':
    loop = 0
    while True:
        print("start loop")
        finished_trains = get_archive_trains(archive_table)
        print(f'{datetime.now().strftime("%H:%M:%S %d/%m/%Y")} - start loop {loop} - {len(finished_trains)}')
        if len(finished_trains) > 80:
            # Specify the number of processes to use in the pool
            num_processes = mp.cpu_count()
            # Split the DataFrame into chunks of size 'chunk_size'
            chunk_size = 1*num_processes

            #chunks = np.array_split(finished_trains, round(len(finished_trains)/chunk_size))
            chunks = np.array_split(finished_trains, 128)
        
            # Create a pool of processes
            pool = Pool(processes=num_processes)

            # Apply the modification function to each chunk in parallel
            modified_chunks = pool.map(process_ids, chunks)

            # Close the pool to free resources
            pool.close()
        print(f'{datetime.now().strftime("%H:%M:%S %d/%m/%Y")} - finished loop {loop} - {len(finished_trains)}')
        loop += 1
        print("*"*50)
        print("-"*50)
        print("*"*50)
        time.sleep(1)
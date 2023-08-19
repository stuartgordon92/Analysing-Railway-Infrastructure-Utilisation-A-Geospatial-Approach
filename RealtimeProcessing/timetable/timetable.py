import json
import os
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import io
import gzip

import awswrangler as wr
import pandas as pd
from boto3.dynamodb.conditions import Attr, Key

from multiprocessing import Pool
import multiprocessing as mp

import sys

import pymongo

def get_cif(b, k, s3):
    response = s3.get_object(Bucket=b, Key=k)
    gzipfile = io.BytesIO(response['Body'].read())
    gzipfile = gzip.GzipFile(fileobj=gzipfile)
    content = gzipfile.read().decode('utf-8')
    return list(dict.fromkeys(content.splitlines()))

def get_tt(cif_in):
    tt = {
        'JsonScheduleV1':[]
    }
    tt_rows = len(cif_in)
    print(cif_in[0])
    for x in range(tt_rows):
        if x % round(tt_rows/10) == 0:
            print(f"{x} / {tt_rows} - {round(x/tt_rows*100)}%")

        json_input = json.loads(cif_in[x])
        json_type = list(json_input.keys())[0]
        json_data = json_input[json_type]
        if json_type == 'JsonScheduleV1':
            tt[json_type].append(json_data)
    return pd.DataFrame(tt['JsonScheduleV1'])

def json_to_series(text):
    keys, values = zip(*[item for item in text.items()])
    return pd.Series(values, index=keys)

def get_scheds(Schedules):
    Schedules['schedule_start_date'] = pd.to_datetime(Schedules['schedule_start_date'])
    Schedules['schedule_end_date'] = pd.to_datetime(Schedules['schedule_end_date'])
    print("length of schedules", len(Schedules))
    return Schedules
    
def trust_time(hhmm, datestr, add_day = False):
    if len(hhmm) == 5:
        hhmm = hhmm[:4] + "30"
    else:
        hhmm = hhmm + "00"
    if add_day:
        datestr = datestr + timedelta(days=1)
    return pd.to_datetime(datestr.strftime('%Y%m%d') + " " + hhmm, format='%Y%m%d %H%M%S')

def org_dest_time(org, dest, date_extract, cutoff_hour = 3):
    if int(org[0:2]) < cutoff_hour:
        org_time = trust_time(org, date_extract, add_day=True)
        dest_time = trust_time(dest, date_extract, add_day=True)
    elif int(dest[0:2]) < int(org[0:2]):
        org_time = trust_time(org, date_extract, add_day=False)
        dest_time = trust_time(dest, date_extract, add_day=True)
    else:
        org_time = trust_time(org, date_extract, add_day=False)
        dest_time = trust_time(dest, date_extract, add_day=False)
    return org_time, dest_time

def berth_est(stnx, plt, arr_dep, smart):
    if arr_dep == 'ARRIVAL':
        events = ['D','B']
        berth_col = 'FROMBERTH'
    if arr_dep == 'DEPARTURE':
        events = ['A','C']
        berth_col = 'TOBERTH'
    if plt is None:      
        try:
            est_berth = "_".join(smart.loc[(smart['STANOX'] == str(stnx)) & (smart['EVENT'].isin(events))][['TD', berth_col]].values[0])
            return est_berth
        except:
            return None
    else:      
        try:
            est_berth = "_".join(smart.loc[(smart['STANOX'] == str(stnx)) & (smart['PLATFORM'] == str(plt)) & (smart['EVENT'].isin(events))][['TD', berth_col]].values[0])
            return est_berth
        except:
            return None

def area_est(stnx, plt, smart):
    try:
        est_areas = smart.loc[(smart['STANOX'] == str(stnx))]['TD'].unique()[0]
        return est_areas
    except:
        return None


def location_message(lo, evt, date_extract, smart, locations_db):

    stanox = int(locations_db[locations_db['Tiploc'] == lo['tiploc_code']]['Stanox'].values[0])
    tiploc = lo['tiploc_code']
    location_name = locations_db[locations_db['Tiploc'] == lo['tiploc_code']]['Name'].values[0]
    location_suffix = lo['tiploc_instance']
    planned_time = None

    if 'line' in lo:
        line = lo['line']
    else:
        line = None
    if 'path' in lo:
        path = lo['path']
    else:
        path = None
    if 'platform' in lo:
        platform = lo['platform']
    else:
        platform = None


    if evt == 'PASS':
        d, a = org_dest_time(lo['pass'], lo['pass'], date_extract, cutoff_hour = 1)
        planned_time = d
        evt = 'DEPARTURE'

    elif evt == 'ORIGIN':
        d, a = org_dest_time(lo['departure'], lo['departure'], date_extract, cutoff_hour = 1)
        planned_time = d
        evt = 'DEPARTURE'

    elif evt == 'DESTINATION':
        d, a = org_dest_time(lo['arrival'], lo['arrival'], date_extract, cutoff_hour = 1)
        planned_time = d
        evt = 'ARRIVAL'
    else:
        d, a = org_dest_time(lo['departure'], lo['arrival'], date_extract, cutoff_hour = 1)
        if evt == 'DEPARTURE':
            planned_time = d
        if evt == 'ARRIVAL':
            planned_time = a
    
    est_berth = berth_est(stanox, platform, evt, smart)
    area_id = area_est(stanox, platform, smart)

    message = {
        'actual_timestamp': None,
        'correction_ind': None,
        'direction_ind': None,
        'event_source': "CIF",
        'event_type': evt,
        'loc_stanox': stanox,
        'location_tiploc': tiploc,
        'location_name': location_name,
        'location_suffix': location_suffix,
        'offroute_ind': None,
        'planned_timestamp': int(datetime.timestamp(planned_time)*1e3),
        'platform': platform,
        'route': None,
        'timetable_variation': "Planned",
        'variation_status': "Planned",
        'line': line,
        'path': path,
        'est_berth': est_berth,
        'area_id': area_id,
    }
    return message


def timetable_processor(in_test, extract_date, smt, locat, ids):
    out_out_tt = []
    rows = len(in_test)
    print(f"starting - chunk - {mp.current_process()._identity[0]}")

    for idx, test in enumerate(in_test):
        try:
            if idx % round(rows/4) == 0:
                print(f"{datetime.now()} - {mp.current_process()._identity[0]} - {idx} / {rows} - {round(idx/rows*100)}%")
            output_tt = []
            no_locations = len(test)
            output_tt.append(location_message(test[0], "ORIGIN", extract_date, smt, locat))
            
            if no_locations > 2:
                for x in range(1, no_locations-1):
                    if test[x]['arrival'] is not None:
                        output_tt.append(location_message(test[x], "ARRIVAL", extract_date, smt, locat))
                    if test[x]['departure'] is not None:
                        output_tt.append(location_message(test[x], "DEPARTURE", extract_date, smt, locat))
                    if test[x]['pass'] is not None:
                        output_tt.append(location_message(test[x], "PASS", extract_date, smt, locat))

            output_tt.append(location_message(test[-1], "DESTINATION", extract_date, smt, locat))
            out_out_tt.append(output_tt)
        except Exception:
            print(f"Issue with index {idx} in list of ids: {ids}")
            output_tt = np.nan
            out_out_tt.append(output_tt)
        
    print(f"ending - chunk = {mp.current_process()._identity[0]} ")
    return out_out_tt

def get_today(sched, ex_date):
    todays_tt = sched[sched.apply(lambda row : True if ((row['schedule_start_date'] <= pd.Timestamp(ex_date) <= row['schedule_end_date']) and (row['schedule_days_runs'][ex_date.weekday()] == '1') and (row['transaction_type'] == 'Create')) else False, axis = 1)]
    print("timetables filtered")

    todays_tt = pd.concat([todays_tt, todays_tt['schedule_segment'].apply(json_to_series)], axis=1).drop(columns=['schedule_segment'])
    print("timetable details expanded")

    print("timetables set to only those running and ordered by stp")
    todays_tt = todays_tt.loc[todays_tt['CIF_stp_indicator'] != 'C']
    todays_tt = todays_tt.loc[todays_tt['train_status'].isin(['F', 'P', 'T', '1', '2', '3'])]

    todays_tt['origin_location'] = todays_tt.apply(lambda row: row['schedule_location'][0]['tiploc_code'], axis=1)
    todays_tt['destination_location'] = todays_tt.apply(lambda row: row['schedule_location'][-1]['tiploc_code'], axis=1)
  
    todays_tt['origin_destination_timestamps'] = todays_tt.apply(lambda row: org_dest_time(row['schedule_location'][0]['departure'], row['schedule_location'][-1]['arrival'], ex_date), axis=1)
    todays_tt['origin_departure_timestamp'] = todays_tt.apply(lambda row: row['origin_destination_timestamps'][0], axis=1)
    todays_tt['destination_arrival_timestamp'] = todays_tt.apply(lambda row: row['origin_destination_timestamps'][1], axis=1)
    todays_tt['duration'] = (todays_tt['destination_arrival_timestamp'] - todays_tt['origin_departure_timestamp']).dt.total_seconds()
    todays_tt['duration'] = todays_tt['duration'].astype(int)
    print("calculated origin and destination")    

    todays_tt['running_date'] = todays_tt['origin_departure_timestamp'].dt.strftime('%Y%m%d')
    todays_tt['stock_type'] = todays_tt['CIF_power_type'] + "_" + todays_tt['CIF_timing_load'] + "_" + todays_tt['CIF_speed']
    
    print(f"timetables id created for no trains, {len(todays_tt)}")
    todays_tt['id'] = todays_tt['CIF_train_uid'] + "_" + todays_tt['signalling_id'] + "_" + todays_tt['running_date']
    todays_tt = todays_tt.sort_values('CIF_stp_indicator', ascending=True)
    todays_tt = todays_tt.drop_duplicates(subset='id', keep='first')



    todays_tt['running_date'] = todays_tt['running_date'].astype(int)

    # todays_tt = todays_tt.loc[todays_tt['atoc_code'].isin(['XR', 'HX', 'GW'])]
    # todays_tt['id'] = todays_tt['CIF_train_uid'] + "_" + todays_tt['signalling_id'] + "_" + ex_date.strftime('%Y%m%d')
    return todays_tt

def enrich_today(tt):
    #extract_date = datetime(2023,5,2).date()
    DATA_PATH = r"C:\Users\stuar\OneDrive - Newcastle University\Year 1\Thesis\Geospatial Model\raw\td_list"

    extract_date = datetime.now().date()
    # DATA_PATH = r"/home/ec2-user/data"

    locations = os.path.join(DATA_PATH, 'locations.json')
    f = open(locations)
    data_location = json.load(f)

    smart = os.path.join(DATA_PATH, 'SMARTExtract.json')
    f = open(smart)
    data_smart = json.load(f)

    smart_db = pd.DataFrame(data_smart['BERTHDATA'])
    locations_db = pd.DataFrame(data_location['Tiplocs'])
    
    tt['planned_timetable'] = timetable_processor(tt['schedule_location'], extract_date, smart_db, locations_db, tt['id'])
    return tt

def export_timetable(final_tt, extract_date, key):

    fname = (key.split("/")[-1]).split(".")[0]
    PWD = r'/home/ec2-user/output'
    EXTRACT_FILE = f"{fname}_extract_{extract_date.strftime('%Y%m%d')}.json"
    extract_file_name = os.path.join(PWD, EXTRACT_FILE)
    final_tt.to_json(extract_file_name, orient="records", indent=4)

def upload_tt_to_dynamo(for_dynamo, table_name):
    items = json.loads(for_dynamo.to_json(orient='records'))
    wr.dynamodb.put_items(items=items, table_name=table_name)
    print(f"{datetime.now()} - tables written")


def dynamo_tt(tt_name):
    dynamodb = boto3.resource('dynamodb', region_name="eu-west-2")
    return dynamodb.Table(tt_name)



def get_expired_tts(date, dynamo):
    response = dynamo.scan(FilterExpression=boto3.dynamodb.conditions.Attr('running_date').eq(date))
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = dynamo.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])

    return data

def write_expired_tts(data, dynamo):
    with dynamo.batch_writer() as writer:
        for item in data:
            writer.put_item(Item=item)

def delete_expired_timetables(keys, dynamo):
    with dynamo.batch_writer() as writer:
        for key in keys:
            writer.delete_item(Key=key)

def archive_old_timetables(day):
    tt_table = dynamo_tt('timetable')
    tt_archive_table = dynamo_tt('timetable_archive')
    try:
        print("Getting old timetables")
        expired_tts = get_expired_tts((day.strftime('%Y%m%d')), tt_table)
        
        ids = []
        for x in range(len(expired_tts)):
            expired_tts[x]['status'] = 'Activation Expired'
            if expired_tts[x]['runningdate'] == day.strftime('%Y%m%d'):
                ids.append({'id': expired_tts[x]['id'], 'origin_departure_timestamp': int(expired_tts[x]['origin_departure_timestamp'])})
        print(f"Expired activations: {len(ids)}")

        print("Uploading old timetables to archive")
        write_expired_tts(expired_tts, tt_archive_table)

        print("Removing old timetables from timetable")
        delete_expired_timetables(ids, tt_table)



    except Exception:
        print('error archiving timetables: ', Exception)


def mongo(url, db):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db_list = mongo_client.list_database_names()
    if db in mongo_db_list:
        print(f"{db} mongo database exists.")
    mongo_db = mongo_client[db]
    
    return mongo_db

def mongo_upload_multi(mongo, tbl_name, df):
    tbllist = mongo.list_collection_names()
    if "tbl_name" in tbllist:
        print(f"{mongo_tbl} table exists.")
    mongo_tbl = mongo[tbl_name]

    x = mongo_tbl.insert_many(df.to_dict('records'))
    return x

def mongo_upload_archive(mongo, tbl_name, data):
    tbllist = mongo.list_collection_names()
    if "tbl_name" in tbllist:
        print(f"{mongo_tbl} table exists.")
    mongo_tbl = mongo[tbl_name]

    x = mongo_tbl.insert_many(data)
    return x

def mongo_find_old(mongo, tbl_name, ex_date):
    tbllist = mongo.list_collection_names()
    if "tbl_name" in tbllist:
        print(f"{mongo_tbl} table exists.")
    mongo_tbl = mongo[tbl_name]

    query = { "running_date": { "$lt": ex_date } }
    resp = mongo_tbl.find(query)
    print(resp)
    return resp

def mongo_delete_old(mongo, tbl_name, ex_date):
    tbllist = mongo.list_collection_names()
    if "tbl_name" in tbllist:
        print(f"{mongo_tbl} table exists.")
    mongo_tbl = mongo[tbl_name]

    query = { "running_date": { "$lt": ex_date } }
    resp = mongo_tbl.delete_many(query)
    print(resp.deleted_count, " documents deleted.")
    return resp

def m_upload(data):
    mongo_database = mongo('', '')
    resp = mongo_upload_multi(mongo_database, 'timetable', data)
    print(resp.inserted_ids)
    return resp

def m_archive(archive_date):
    mongo_database = mongo('', '')
    archive_data = mongo_find_old(mongo_database, 'timetable', archive_date)
    if len(archive_data) < 0:
        resp = mongo_upload_archive(mongo_database, 'timetable-archive', archive_data)
        delete_data = mongo_delete_old(mongo_database, 'timetable', archive_date)
        return resp
    else:
        return None



if __name__ == '__main__':
    

    ##### TEST Date ####
    #extract_date = datetime(2023,5,2).date()
    #key = 'nrodv1.1/topic/CIF_ALL_FULL_DAILY/toc_full_daily_cif_20230502.json.gz'


    s3 = boto3.client('s3')
    extract_date = datetime.now().date()
    yesterday = extract_date - timedelta(days=1)

    bucket = 't2-data-132515-123413-1251512'
    key = f"message-backup/nrod/CIF_ALL_FULL_DAILY/date={extract_date.strftime('%Y%m%d')}/toc_full_daily_cif_{extract_date.strftime('%Y%m%d')}.gz"

    print(datetime.now())
    print("Get CIF, key")
    cif = get_cif(bucket, key, s3)
    
    print(datetime.now())
    print("Get timetable")
    timetable = get_tt(cif)
    
    print(datetime.now())
    print("Get schedules")
    scheds = get_scheds(timetable)
    
    print(datetime.now())
    print("Get todays timetable")
    todays_timetable = get_today(scheds, extract_date)

    print(datetime.now())
    print("Erich todays timetable")

    # Specify the number of processes to use in the pool
    num_processes = mp.cpu_count()
    # Split the DataFrame into chunks of size 'chunk_size'
    chunk_size = 10*num_processes
    # chunks = [todays_timetable[i:i+chunk_size] for i in range(0, todays_timetable.shape[0], chunk_size)]
    chunks = np.array_split(todays_timetable, round(todays_timetable.shape[0]/chunk_size))
   
    # Create a pool of processes
    pool = Pool(processes=num_processes)

    # Apply the modification function to each chunk in parallel
    modified_chunks = pool.map(enrich_today, chunks)

    # Close the pool to free resources
    pool.close()

    todays_timetable_mp = pd.concat(modified_chunks)

    todays_timetable_mp = todays_timetable_mp[['id', 'CIF_train_uid', 'signalling_id', 'running_date',
        'atoc_code', 'schedule_start_date', 'schedule_end_date', 'CIF_stp_indicator',
        'CIF_train_service_code', 'stock_type', 
        'origin_location', 'origin_departure_timestamp', 'destination_location', 'destination_arrival_timestamp', 'duration', 'planned_timetable'
        ]].rename(columns={'id': '_id'})
    
    print(f"{datetime.now()} - Error timetable IDs - {len(todays_timetable_mp[todays_timetable_mp['planned_timetable'].isna()])}")
    print(todays_timetable_mp[todays_timetable_mp['planned_timetable'].isna()])

    print(f"{datetime.now()} - Timetable Processing complete")
    #export_timetable(todays_timetable_mp, extract_date, key)

    # Mongo Connections
    print(f"{datetime.now()} - Timetables uploading to mongo")
    x = m_upload(todays_timetable_mp)
    print(f"{datetime.now()} - Timetables uploaded to mongo = {len(x.inserted_ids)}")

    print(f"{datetime.now()} - Archive old timetables")
    arc = m_archive(int(extract_date.strftime('%Y%m%d')))
    print(f"{datetime.now()} - Old Timetables archived = {len(arc.inserted_ids) if arc is not None else 'None Archived'}")

    # print(f"{datetime.now()} - Shutting down instance")
    # os.system('shutdown now -h')


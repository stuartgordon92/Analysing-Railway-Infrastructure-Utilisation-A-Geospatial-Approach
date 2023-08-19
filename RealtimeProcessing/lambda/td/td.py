import json
import boto3
from datetime import datetime
import base64
import os
import pymongo

MONGO_URL=os.getenv('mongo_url')
MONGO_DB=os.getenv('mongo_db')

RUNNING_TABLE_NAME=os.getenv('running_table_name')
BERTH_TABLE_NAME=os.getenv('berth_table_name')

def mongo(url, db, table):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db = mongo_client[db]
    mongo_tbl = mongo_db[table]
    return mongo_tbl


def assign_td_message(td_c_message, running_mongo, berths_mongo):
    running_train = get_train_running_tt(td_c_message, running_mongo)
    if running_train is not None:
        running_train = update_movement_table_td(td_c_message, running_train, running_mongo, berths_mongo)
    else:
        print(f"ERROR - TD - No trains in running table - {td_c_message['descr']}")


def get_tt_headcodes(headcode, mongo):
    key = {'signalling_id': headcode}
    try:
        response = mongo.find(key)
        responses = list(response)
        return responses 
    except Exception as e: 
        print("Exception - Error calling mongodb - get item in tt table")
        print(f"Mongo Exception - Function: get_tt_headcodes - ID: {key} - {e}")
        return None

def get_berth_geo(berth, mongo):
    try:
        response = mongo.find({'_id': berth}, {"_id": 0, "geometry": 1})
        responses = list(response)
        return responses 
    except Exception as e: 
        print("Exception - Error calling mongodb - get item in tt table")
        print(f"Mongo Exception - Function: get_berth_geo - ID: {berth} - {e}")
        return None






def deep_search_trains(td_i, possible_trains):
    print(f"Deep search - {td_i['descr']}")
    for x in possible_trains:
        print(f"--- Deep search possibles - {x['_id']}")
    possible_areas = []
    for x in range(len(possible_trains)):
        if 'current_location_number' in possible_trains[x] and possible_trains[x]['current_location_number'] is not None:
            loc_num = int(possible_trains[x]['current_location_number'])
            current_area = possible_trains[x]['actual_timetable'][loc_num]['area_id'] if 'area_id' in possible_trains[x]['actual_timetable'][loc_num] else None
            if loc_num > 0:
                prev_area = possible_trains[x]['actual_timetable'][loc_num]['area_id'] if 'area_id' in possible_trains[x]['actual_timetable'][loc_num] else None
            else: 
                prev_area = None
            if loc_num < (len(possible_trains[x]['actual_timetable'])-1):
                next_area = possible_trains[x]['actual_timetable'][loc_num]['area_id'] if 'area_id' in possible_trains[x]['actual_timetable'][loc_num] else None
            else: 
                next_area = None
            area_id_list = [prev_area, current_area, next_area]
            area_id_set = set(area_id_list)
        else:
            area_id_set = set()
        possible_areas.append(list(area_id_set))

    match = False
    if any(td_i['area_id'] in sublist for sublist in possible_areas):
        for ind, x in enumerate(possible_areas):
            if td_i['area_id'] in x:
                match_train = possible_trains[ind]
                match = True
    if match:
        print(f"--- Matched Train: {match_train['_id']}")
        return match_train
    else:
        for x in possible_trains:
            print(f"--- Deep search - Couldnt find a unique match for {td_i['descr']} in following:")
            print(f"--- Deep search - {x['_id']}")
        return None

def get_train_running_tt(td_i, running_dyna):
    tts = get_tt_headcodes(td_i['descr'], running_dyna)

    if len(tts) == 1:
        return tts[0]
    elif len(tts) > 1:
        print(f"Scan returned no of items: {len(tts)}")
        single_tt = deep_search_trains(td_i, tts)
        return single_tt
    else:
        return None
        
def update_movement_table_td(td_msg, running_data, mongo, berths_mong):
    item = update_running_tt(td_msg, running_data, berths_mong)
    # print(f"Updating {item['_id']} - {item['signalling_id']} - {item['current_step']}")
    try:
        key = {'_id': item['_id']}
        mongo.replace_one(key, item)
    except Exception as e: 
        print("Exception - Error calling mongodb - update item in running table")
        print(f"Mongo Exception - Function: update_movement_table_td - ID: {item['_id']} - {e}")


def update_running_tt(td, running_tt, berths):
    running_tt['last_updated'] = datetime.fromtimestamp(int(td['time'])/1e3).strftime('%Y/%m/%d, %H:%M:%S')
    running_tt['last_updated_TTL'] = int(td['time'])
    running_tt['current_berth'] = td['to'] if 'to' in td else td['from'] if 'from' in td else None
    running_tt['current_step'] = td['step'] if 'step' in td else None
    running_tt['current_area_id'] = td['area_id']
    running_tt['last_step_time'] = int(td['time'])
    td['timetable_variation'] = running_tt['current_timetable_variation'] if 'current_timetable_variation' in running_tt else None
    if 'td_messages' in running_tt:
        running_tt['td_messages'].append(td)
    else:
        running_tt['td_messages'] = [td]

    berth_geo_loc = get_berth_geo(running_tt['current_berth'], berths)
    if len(berth_geo_loc) == 1:
        running_tt['geo_location'] = berth_geo_loc[0]

    return running_tt

def lambda_handler(event, context):  
    running_table = mongo(MONGO_URL, MONGO_DB, RUNNING_TABLE_NAME)
    berth_table = mongo(MONGO_URL, MONGO_DB, BERTH_TABLE_NAME)
    for queues in event['rmqMessagesByQueue']:
        for message in event['rmqMessagesByQueue'][queues]:
            data = json.loads(base64.b64decode(message['data']).decode('ascii'))
            # for x in data:
                # print (x)
            if 'descr' in data:
                assign_td_message(data, running_table, berth_table)




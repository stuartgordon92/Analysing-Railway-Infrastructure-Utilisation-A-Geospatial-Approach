import json
import base64
import boto3
from datetime import datetime
import os
import pymongo

MONGO_URL=os.getenv('mongo_url')
MONGO_DB=os.getenv('mongo_db')

TIMETABLE_TABLE_NAME=os.getenv('timetable_table_name')
ACTIVATED_TABLE_NAME=os.getenv('activated_table_name')
RUNNING_TABLE_NAME=os.getenv('running_table_name')
ARCHIVE_TABLE_NAME=os.getenv('archive_table_name')

def lambda_handler(event, context):
    # print(event)
    tt_table = mongo(MONGO_URL, MONGO_DB, TIMETABLE_TABLE_NAME)
    activation_table = mongo(MONGO_URL, MONGO_DB, ACTIVATED_TABLE_NAME)
    running_table = mongo(MONGO_URL, MONGO_DB, RUNNING_TABLE_NAME)
    archive_table = mongo(MONGO_URL, MONGO_DB, ARCHIVE_TABLE_NAME)


    for queues in event['rmqMessagesByQueue']:
        for message in event['rmqMessagesByQueue'][queues]:
            msg = json.loads(base64.b64decode(message['data']).decode('ascii'))
            msg_type = msg['header']['msg_type']
            print(f"Message Type: {msg_type} - {msg['body']['train_id']}")
            if msg_type == "0001":

                print("Get data from timetable table")
                tt = get_tt_data(msg, tt_table)
                if tt is not None:
                    print("Removing record from timetable table")
                    remove_timetable_record(msg, tt_table)                 
                    activated_tt = activation_process(msg, tt)
                    print("Put data into activated table")
                    put_activated_data(activated_tt, activation_table)
                else:
                    print(f"Could not get tt data for {msg['body']['train_id']}")

            elif msg_type == "0002":
                print("Cancelled Train")

                if msg['body']['canx_type'] == 'AT ORIGIN' or msg['body']['canx_type'] == 'ON CALL':
                    cancelled_train = get_activated_data(msg, activation_table)
                    remove_activated_record(msg, activation_table)
                elif msg['body']['canx_type'] == 'EN ROUTE' or msg['body']['canx_type'] == 'OUT OF PLAN':
                    cancelled_train = get_running_data(msg, running_table)
                    remove_running_record(msg, running_table)
                
                if cancelled_train is not None:
                    cancelled_train = update_cancelled_train(msg, cancelled_train)
                    put_finished_tt(cancelled_train, archive_table)
                else:
                    print(f"Could not get train data for cancelled {msg['body']['train_id']}")

            elif msg_type == "0003":
                print(f"{msg['body']['train_id']} - Find in running timetable")
                tt_running = get_running_data(msg, running_table)
                if tt_running is None:

                    print(f"{msg['body']['train_id']} - Find TT in activation table")
                    tt_activated = get_activated_data(msg, activation_table)
                    if tt_activated is None:
                        pass
                        # print(f"{msg['body']['train_id']} - Cant find train activattion, creating running train")
                        # running_tt_data = create_running_tt_from_scratch(msg)
                        # put_running_tt(running_tt_data, running_table)

                    else:
                        print(f"{msg['body']['train_id']} - Found activated train, moving to running")
                        remove_activated_record(msg, activation_table)
                        running_tt_data = create_running_tt_from_activation(msg, tt_activated)
                        put_running_tt(running_tt_data, running_table)
                else:
                    print(f"{msg['body']['train_id']} - Running train moved, updating data")
                    running_tt_data = update_running_tt(msg, tt_running)
                    update_movement_table(running_tt_data, running_table)

                if msg['body']['train_terminated'] == "true" and tt_running is not None:
                    print(f"{msg['body']['train_id']} - Train terminating, moving to finished")
                    finished_tt_data = create_finished_tt_from_running(msg, tt_running)
                    put_finished_tt(finished_tt_data, archive_table)
                    remove_running_record(msg, running_table)
            else:
                print("other message type")


def update_cancelled_train(canx_msg, canx_tt):
    canx_tt['status'] = f"Cancelled - {canx_msg['body']['canx_type']}"
    canx_tt['cancelled'] = True
    canx_tt['cancelled_timestamp'] = int(canx_msg['body']['canx_timestamp'])
    canx_tt['cancelled_reason'] = canx_msg['body']['canx_reason_code']

    for x in range(len(canx_tt['actual_timetable'])):
        if int(canx_tt['actual_timetable'][x]['loc_stanox']) == int(canx_msg['body']['loc_stanox']):
            loc_num = x

    if 'loc_num' in locals():
        canx_tt['actual_timetable'] = canx_tt['actual_timetable'][0:(loc_num+1)]
    
    return canx_tt


def create_finished_tt_from_running(mvt_msg, finished_tt):
    finished_tt['finished_location'] = finished_tt['current_location'] if 'current_location' in finished_tt else mvt_msg['body']['loc_stanox']
    finished_tt['finished_running_timestamp'] = finished_tt['last_updated_TTL'] if 'last_updated_TTL' in finished_tt else int(mvt_msg['header']['msg_queue_timestamp'])
    if 'start_running_timestamp' in finished_tt:
        finished_tt['running_duration'] = finished_tt['finished_running_timestamp'] - finished_tt['start_running_timestamp']
    else:
        finished_tt['running_duration'] = None
    finished_tt['status'] = 'Finished'
    finished_tt['finished_variation_status'] = finished_tt['current_variation_status'] if 'current_variation_status' in finished_tt else None
    finished_tt['finished_timetable_variation'] = finished_tt['current_timetable_variation'] if 'current_timetable_variation' in finished_tt else None
    if 'id' not in finished_tt:
        finished_tt['id'] = finished_tt['train_id'] + "_" + str(finished_tt['running_date'])
    finished_tt['_id'] = finished_tt['train_id'] + "_" + str(finished_tt['running_date'])
    return finished_tt

def put_finished_tt(tt_input, mongo):
    try:
        mongo.insert_one(tt_input)
    except Exception as e: 
        print("Exception - Error calling mongodb - put item in running table")
        print(f"Mongo Exception - Function: put_finished_tt - ID: {tt_input['_id']} - {e}")

def remove_running_record(tt_input, mongo):
    id = tt_input['body']['train_id']
    key = {'_id': id}
    try:
        response = mongo.delete_one(key)
    except Exception as e: 
        print("Exception - Error calling mongodb - remove item in running table")
        print(f"Mongo Exception - Function: remove_running_record - ID: {key} - {e}")

def update_movement_table(item, mongo):
    try:
        key = {'_id': item['_id']}
        mongo.replace_one(key, item)
    except Exception as e: 
        print("Exception - Error calling mongodb - update item in running table")
        print(f"Mongo Exception - Function: update_movement_table - ID: {item['_id']} - {e}")


def update_running_tt(movement_msg, running_tt):
    running_tt['last_updated'] = datetime.fromtimestamp(int(movement_msg['header']['msg_queue_timestamp'])/1e3).strftime('%Y/%m/%d, %H:%M:%S')
    running_tt['last_updated_TTL'] = int(movement_msg['header']['msg_queue_timestamp'])
    running_tt['actual_timetable'], running_tt['current_location_number'] = update_actual_timetable(movement_msg['body'], running_tt['actual_timetable'])
    running_tt['current_location'] = movement_msg['body']['loc_stanox']
    running_tt['current_variation_status'] = movement_msg['body']['variation_status']
    running_tt['current_timetable_variation'] = movement_msg['body']['timetable_variation']
    running_tt['movement_messages'].append(movement_msg['body'])
    return running_tt


def create_running_tt_from_activation(movement_msg, activation_tt):
    activation_tt['last_updated'] = datetime.fromtimestamp(int(movement_msg['header']['msg_queue_timestamp'])/1e3).strftime('%Y/%m/%d, %H:%M:%S')
    activation_tt['last_updated_TTL'] = int(movement_msg['header']['msg_queue_timestamp'])
    activation_tt['start_running_timestamp'] = int(movement_msg['header']['msg_queue_timestamp'])
    activation_tt['actual_timetable'], activation_tt['current_location_number'] = update_actual_timetable(movement_msg['body'], activation_tt['actual_timetable'])
    activation_tt['current_location'] = movement_msg['body']['loc_stanox']
    activation_tt['current_variation_status'] = movement_msg['body']['variation_status']
    activation_tt['current_timetable_variation'] = movement_msg['body']['timetable_variation']
    activation_tt['movement_messages'] = [movement_msg['body']]
    activation_tt['status'] = 'Running'
    return activation_tt

def update_actual_timetable(mvt_message_body, act_tt):
    current_location_num = None
    for x in range(len(act_tt)):
        if 'planned_timestamp' in mvt_message_body and 'planned_timestamp' in act_tt[x]:
            if int(mvt_message_body['loc_stanox']) == int(act_tt[x]['loc_stanox']) and int(mvt_message_body['planned_timestamp']) == int((act_tt[x]['planned_timestamp'])):
                act_tt[x]['actual_timestamp'] = mvt_message_body['actual_timestamp']
                act_tt[x]['correction_ind'] = mvt_message_body['correction_ind']
                act_tt[x]['direction_ind'] = mvt_message_body['direction_ind'] if 'direction_ind' in mvt_message_body else None
                act_tt[x]['event_type'] = mvt_message_body['event_type']
                act_tt[x]['offroute_ind'] = mvt_message_body['offroute_ind']
                act_tt[x]['platform'] = mvt_message_body['platform'] if 'platform' in mvt_message_body else None
                act_tt[x]['route'] = mvt_message_body['route'] if 'route' in mvt_message_body else None
                act_tt[x]['line'] = mvt_message_body['line_ind'] if 'line_ind' in mvt_message_body else None
                act_tt[x]['timetable_variation'] = mvt_message_body['timetable_variation']
                act_tt[x]['variation_status'] = mvt_message_body['variation_status']
                current_location_num = x
        else:
            print(f"WARNING - Issue with planned timestamp not in one of: {mvt_message_body}, {act_tt[x]}")
    return act_tt, current_location_num

def create_running_tt_from_scratch(tt_input):
    return {
    "_id": tt_input['body']['train_id'],
	"train_id": tt_input['body']['train_id'],
	"activated_by": "AWS - no tt",
	"activated_timestamp": int(tt_input['header']['msg_queue_timestamp']),
	"actual_timetable": [
        {
            "actual_timestamp": int(tt_input['body']['actual_timestamp']),
			"correction_ind": tt_input['body']['correction_ind'],
			"direction_ind": tt_input['body']['direction_ind'] if 'direction_ind' in tt_input['body'] else None,
			"est_berth": None,
			"event_source": tt_input['body']['event_source'],
			"event_type": tt_input['body']['event_type'],
			"line": tt_input['body']['line_ind'] if 'line_ind' in tt_input['body'] else None,
			"location_name": None,
			"location_suffix": None,
			"location_tiploc": None,
			"loc_stanox": tt_input['body']['loc_stanox'],
			"offroute_ind": tt_input['body']['offroute_ind'],
			"path": None,
			"planned_timestamp": int(tt_input['body']['planned_timestamp']) if 'planned_timestamp' in tt_input['body'] else None,
			"platform": tt_input['body']['platform'] if 'platform' in tt_input['body'] else None,
			"route": tt_input['body']['route'] if 'route' in tt_input['body'] else None,
			"timetable_variation": tt_input['body']['timetable_variation'],
			"variation_status": tt_input['body']['variation_status']
        }
    ],
	"atoc_code": None,
	"CIF_stp_indicator": None,
	"CIF_train_service_code": tt_input['body']['train_service_code'],
	"CIF_train_uid": None,
	"destination_arrival_timestamp": None,
	"destination_location": None,
	"duration": None,
	"id": None,
	"last_updated": datetime.fromtimestamp(int(tt_input['header']['msg_queue_timestamp'])/1e3).strftime('%Y/%m/%d, %H:%M:%S'),
	"last_updated_TTL": int(tt_input['header']['msg_queue_timestamp']),
	"origin_departure_timestamp": None,
	"origin_location": None,
	"planned_timetable": [],
    "movement_messages": [tt_input['body']],
	"running_date": datetime.fromtimestamp(int(tt_input['header']['msg_queue_timestamp'])/1e3).strftime('%Y%m%d'),
	"schedule_end_date": None,
	"schedule_start_date": None,
	"signalling_id": tt_input['body']['train_id'][2:6],
	"status": "Running",
	"stock_type": None,
    "current_location": tt_input['body']['loc_stanox'],
    "current_variation_status": tt_input['body']['variation_status'],
    "current_timetable_variation": tt_input['body']['timetable_variation']
}

def put_running_tt(tt_input, mongo):
    try:
        mongo.insert_one(tt_input)
    except Exception as e: 
        print("Exception - Error calling mongodb - put item in running table")
        print(f"Mongo Exception - Function: put_running_tt - ID: {tt_input['_id']} - {e}")



def remove_activated_record(tt_input, mongo):
    id = tt_input['body']['train_id']
    key = {'_id': id}
    try:
        response = mongo.delete_one(key)
    except Exception as e: 
        print("Exception - Error calling mongodb - remove item in activated table")
        print(f"Mongo Exception - Function: remove_activated_record - ID: {key} - {e}")


def get_running_data(tt_input, mongo):
    id = tt_input['body']['train_id']
    key = {'_id': id}
    try:
        response = mongo.find_one(key)
        return response  
    except Exception as e: 
        print("Exception - Error calling mongodb - get item in running table")
        print(f"Mongo Exception - Function: get_running_data - ID: {key} - {e}")



def get_activated_data(tt_input, mongo):
    id = tt_input['body']['train_id']
    key = {'_id': id}
    try:
        response = mongo.find_one(key)
        return response    
    except Exception as e: 
        print("Exception - Error calling mongodb - get item in activation table")
        print(f"Mongo Exception - Function: get_activated_data - ID: {key}" - {e})


def get_tt_data(tt_input, mongo):
    id = get_id(tt_input)
    key = {'_id': id}
    try:
        response = mongo.find_one(key)
        return response   
    except Exception as e: 
        print("Exception - Error calling mongodb - get item in tt table")
        print(f"Mongo Exception - Function: get_tt_data - ID: {key} - {e}")


def activation_process(activation_msg, tt_extract):
    tt_extract['actual_timetable'] = tt_extract['planned_timetable']
    tt_extract['_id'] = activation_msg['body']['train_id']
    tt_extract['train_id'] = activation_msg['body']['train_id']
    tt_extract['last_updated'] = datetime.fromtimestamp(int(activation_msg['header']['msg_queue_timestamp'])/1e3).strftime('%Y/%m/%d, %H:%M:%S')
    tt_extract['activated_by'] = activation_msg['body']['train_call_type'] 
    tt_extract['activated_timestamp'] = int(activation_msg['header']['msg_queue_timestamp'])
    tt_extract['last_updated_TTL'] = int(activation_msg['header']['msg_queue_timestamp'])
    tt_extract['status'] = "Activated"
    return tt_extract

def remove_timetable_record(tt_input, mongo):
    id = get_id(tt_input)
    key = {'_id': id}
    try:
        response = mongo.delete_one(key)
    except Exception as e: 
        print("Exception - Error calling mongodb - remove item in tt table")
        print(f"Mongo Exception - Function: remove_timetable_record - ID: {key - {e}}")


def put_activated_data(tt_input, mongo):
    try:
        mongo.insert_one(tt_input)
    except Exception as e: 
        print("Exception - Error calling mongodb - put item in activated table")
        print(f"Mongo Exception - Function: put_activated_data - ID: {tt_input['_id']} - {e}")


def get_id(input_data):
    uid = input_data['body']['train_uid']
    headcode = input_data['body']['schedule_wtt_id'][:-1]
    running_date = "".join(input_data['body']['tp_origin_timestamp'].split("-"))
    return "_".join([uid, headcode, running_date])



def mongo(url, db, table):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db = mongo_client[db]
    mongo_tbl = mongo_db[table]
    return mongo_tbl
from datetime import datetime
import os
import pymongo

MONGO_URL=os.getenv('mongo_url')
MONGO_DB=os.getenv('mongo_db')

ACTIVATED_TABLE_NAME=os.getenv('activated_table_name')
RUNNING_TABLE_NAME=os.getenv('running_table_name')
ARCHIVE_TABLE_NAME=os.getenv('archive_table_name')

def mongo(url, db, table):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db = mongo_client[db]
    mongo_tbl = mongo_db[table]
    return mongo_tbl


def get_expired_activations(threshold, mongo):
    key = {'activated_timestamp': {"$lt": threshold}}
    try:
        response = mongo.find(key)
        responses = list(response)
        return responses
    except Exception as e: 
        print("Exception - Error calling mongodb - get item in tt table")
        print(f"Mongo Exception - Function: get_expired_activations - ID: {key} - {e}")
        return []

def get_expired_running(threshold, mongo):
    key = {'last_updated_TTL': {"$lt": threshold}}
    try:
        response = mongo.find(key)
        responses = list(response)
        return responses
    except Exception as e: 
        print("Exception - Error calling mongodb - get item in tt table")
        print(f"Mongo Exception - Function: get_expired_activations - ID: {key} - {e}")
        return []

def delete_activatons(keys, mongo):
    try:
        response = mongo.delete_many({'_id': {"$in": keys}})
        print(response.deleted_count, " documents deleted.")
    except Exception as e: 
        print("Exception - Error calling mongodb - get item in tt table")
        print(f"Mongo Exception - Function: get_expired_activations - ID: {keys} - {e}")

def archive_tts(data, mongo):
    try:
        response = mongo.insert_many(data)
        print(response.inserted_ids, " documents inserted.")
    except Exception as e: 
        print("Exception - Error calling mongodb - get item in tt table")
        print(f"Mongo Exception - Function: get_expired_activations - ID: {data} - {e}")

def lambda_handler(event, context):
    activations_table = mongo(MONGO_URL, MONGO_DB, ACTIVATED_TABLE_NAME)
    running_table = mongo(MONGO_URL, MONGO_DB, RUNNING_TABLE_NAME)
    archive_table = mongo(MONGO_URL, MONGO_DB, ARCHIVE_TABLE_NAME)

    timestamp = int(datetime.now().timestamp()*1e3)
    buffer = 28800*1e3  
    cuttoff = int(timestamp - buffer)

    print("Get expired activations")
    expired_activations = get_expired_activations(cuttoff, activations_table)
    print(f"Found {len(expired_activations)} expired runnings")
    if len(expired_activations) > 0:
        ids = []
        for x in range(len(expired_activations)):
            expired_activations[x]['status'] = 'Activation Expired'
            if 'id' not in expired_activations[x]:
                expired_activations[x]['_id'] = expired_activations[x]['train_id'] + "_" + datetime.now().strftime('%Y%m%d')
            ids.append(expired_activations[x]['train_id'])
        print(f"Expired activations: {ids}")
        
        print("Sending to archive table")
        archive_tts(expired_activations, archive_table)
        
        print("Deleting from activation table")
        delete_activatons(ids, activations_table)
    
    
    # ####### Running
    timestamp = int(datetime.now().timestamp()*1e3)
    buffer = 3600*1e3    
    cuttoff = int(timestamp - buffer)

    print("Get expired runnings")
    expired_runnings = get_expired_running(cuttoff, running_table)
    print(f"Found {len(expired_runnings)} expired runnings")
    if len(expired_runnings) > 0:
        ids = []
        for x in range(len(expired_runnings)):
            expired_runnings[x]['status'] = 'Running Train Expired'
            if 'id' not in expired_runnings[x] or expired_runnings[x]['id'] == None:
                expired_runnings[x]['_id'] = expired_runnings[x]['train_id'] + "_" + datetime.now().strftime('%Y%m%d')
            ids.append(expired_runnings[x]['train_id'])
        print(f"Expired runnings: {ids}")

        print("Sending to archive table")
        archive_tts(expired_runnings, archive_table)

        print("Deleting from activation table")
        delete_activatons(ids, running_table)

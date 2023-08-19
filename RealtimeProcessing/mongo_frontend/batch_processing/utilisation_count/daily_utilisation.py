import os
import pickle
import pymongo
import geopandas as gpd
from datetime import datetime

MONGO_URL_FE=os.getenv'mongo_URL_FE'
MONGO_DB=os.getenv'mongo_DB'
ACTUAL_TABLE_NAME='utilisation'
PLAN_TABLE_NAME='utilisation_v2'


def get_daily_data(mongo, process_date):
    query = {'running_date': int(process_date)}
    resp = list(mongo.find(query))
    return resp

def mongo(url, db, table):
    mongo_client = pymongo.MongoClient(f"mongodb://{url}:27017/")
    mongo_db = mongo_client[db]
    mongo_tbl = mongo_db[table]
    return mongo_tbl

data_load_path = 'NetworkModelWGS84_v1.pkl'
with open(data_load_path, 'rb') as file:
    # Call load method to deserialze
    model = pickle.load(file)
networkmodel_wgs84 = model
networkmodel_wgs84 = networkmodel_wgs84.set_index('ASSETID')

ran = range(20230711,20230724)
tables = [ACTUAL_TABLE_NAME, PLAN_TABLE_NAME]

for tab in tables:
    util_table = mongo(MONGO_URL_FE, MONGO_DB, tab)

    extract_date = [str(x) for x in ran]
    extract_date_names = [f'{tab}_{str(x)}' for x in ran]
    for asd in extract_date:
        print(asd)
        day = get_daily_data(util_table, asd)
        counts = {}
        for d in day:
            if len(d['elements']) > 0:
                for x in d['elements']:
                    if x in counts:
                        counts[x] += 1
                    else:
                        counts[x] = 1

        networkmodel_wgs84[f'{tab}_{str(asd)}'] = counts
    networkmodel_wgs84[f'{tab}_total'] = networkmodel_wgs84[extract_date_names].sum(axis=1)


extract_date = [str(x) for x in ran]
for t in extract_date:
    networkmodel_wgs84[f'{t}_diff'] = networkmodel_wgs84[f'{tables[1]}_{str(t)}'] - networkmodel_wgs84[f'{tables[0]}_{str(t)}']
networkmodel_wgs84['diff_total'] = networkmodel_wgs84[f'{tables[1]}_total'] - networkmodel_wgs84[f'{tables[0]}_total']

extract_cols = ['ELR', 'TRID', 'Mileage_From', 'Mileage_To', 'geometry', f'{tables[0]}_total', f'{tables[1]}_total', 'diff_total']
tab_date_names = [f'{tables[0]}_{str(x)}' for x in ran] + [f'{tables[1]}_{str(x)}' for x in ran] + [f'{str(x)}_diff' for x in ran]

extra = extract_cols + tab_date_names

date_dict = {item: 0 for item in tab_date_names}

output = networkmodel_wgs84[extra].fillna(date_dict).reset_index(drop=False)#.rename(columns={'counts': extract_date, 'ASSETID': '_id'})

output_path = f'output_final_combined.geojson'
output.to_file(output_path, driver="GeoJSON") 
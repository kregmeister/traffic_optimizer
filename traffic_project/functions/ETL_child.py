import psycopg2
import s3fs
import os
import ast
import json
import numpy as np
import pandas as pd

def feature_engineering(tables, fs, s3_file, connection, cursor):
    # Entries to the CSV cannot be serialized like in PostgreSQL, so this is done manually
    vid = 0
    
    # Each of the 96 tables is appended to this DataFrame, which will be written to a CSV table in S3
    all_rows = []
    
    vid = 0
    for table in tables:
        # Convert table name (modified timestamp) into a datetime object so queries can order by 
        dt_str = table.replace("dt_", "").replace("_", "-", 2).replace("_", " ", 1).replace("_", ":")
        
        # Get data from each table
        cursor.execute(f"""SELECT * FROM {table};""")
        data = cursor.fetchall()
        velocities = [x[1] for x in data]
        vehicle_types = [t[0] for t in data]
        accident_bools = [t[-1] for t in data]
        
        # Finds all unique vehicle types to encode
        unique_vtypes, encoded_integers = np.unique(vehicle_types, return_inverse=True)
        
        # Calculates an estimated travel time from a vehicle's velocity
        travel_times = (60 / np.array(velocities)) * 5
        
        # Collect rows from table to be re-formatted to Pandas DataFrame
        rows = [
            {
                "id": i + 1 + vid,
                "datetime": dt_str,
                "vehicle_type": vehicle_types[i],
                "vehicle_encoding": encoded_integers[i],
                "velocity": velocities[i],
                "travel_time": travel_times[i],
                "accident": accident_bools[i]
            }
            for i in range(len(data))
        ]
        
        vid += len(data)
        all_rows.extend(rows)
        
    # Create DataFrame from list of table dictionaries
    df = pd.DataFrame(all_rows)
        
    # Write DataFrame to S3 as 
    with fs.open(s3_file, 'w') as f:
        df.to_csv(f, index=False)

def lambda_handler(event, context):
    tables = ast.literal_eval(event.get('tables')) # Converts stringed list of tables back to list (passed from parent function)
    date = event.get('date')
    
    # Connects to RDS PostgreSQL DB
    conn = psycopg2.connect(f"dbname=postgres user=postgres password={os.environ.get("POSTGRES_PW")} host={os.environ.get("POSTGRES_HOST")}, port=5432")
    cur = conn.cursor()
    
    # Establishes file system of S3 bucket and path of data CSV
    fs = s3fs.S3FileSystem()
    traffic_csv = f's3://cjytc.other/traffic_project/data/june/highway_x_june_{date}.csv'
    
    feature_engineering(tables, fs, traffic_csv, conn, cur)
    
    conn.close()
    print("Complete!")

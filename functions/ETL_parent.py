import json
import os
import boto3
from botocore.config import Config
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from time import sleep

def all_rds_tables(cursor, connection):
    # Gets all table names (should be 2880 of them)
    cursor.execute("""SELECT table_name
                FROM information_schema.tables
                WHERE table_schema='public'
                AND table_type='BASE TABLE'
                ORDER BY table_name;""")
    tables = cursor.fetchall()
    tables = [t[0] for t in tables]
    print(f"Number of tables in DB: {len(tables)}")
    
    return tables

def lambda_handler(event, context):
    # Sets environment variables (for legibility)
    endpoint = os.environ.get('POSTGRES_ENDPOINT')
    pw = os.environ.get('POSTGRES_MASTER_PW')
    
    # Connects to RDS using password auth
    connection = psycopg2.connect(f"dbname=postgres user=postgres password={pw} host={endpoint} port=5432")
    cursor = connection.cursor()
    
    # Gets all table names in database
    tables = all_rds_tables(cursor, connection)
    print("Table names acquired.")
    
    connection.close()
    
    print("Starting async invocation...")
    my_config = Config(
        retries = {
            'max_attempts': 3,
            'mode': 'standard'
        },
        connect_timeout=50,
        read_timeout=70,
        max_pool_connections=50  # Increase pool size
        )
    
    client = boto3.client('lambda', config=my_config)
    print(f"Lambda client initialized at {datetime.now()}")
    
    # Function to invoke Lambda asynchronously
    def invoke_lambda(tables, date):
        payload = json.dumps(
            {
            'tables': tables,
            'date': date
            }
        ) # Converts list of tables to JSON string so it can be passed to child function
    
        response = client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:992382556703:function:traffic_data_preperation',
            InvocationType='Event', # Asynchronous invocation
            Payload=payload
        )
        return json.load(response['Payload'])
    
    # This list comprehension splits the ordered table names into 30 segments (days), 96 tables per day
    index_groupings = range(0, len(tables), (len(tables) // 30))
    csv_table_groupings = [tables[i:i+96] for i in index_groupings]
    
    # Use ThreadPoolExecutor to manage concurrent invocations
    with ThreadPoolExecutor(max_workers=15) as executor:
        # Passes respective day and its associated table names to async child invocation function
        day_nums = [f"0{i}" if i < 10 else str(i) for i in range(1, 31)]
        futures = [executor.submit(invoke_lambda, str(tables), str(day)) for tables, day in zip(csv_table_groupings, day_nums)] 

    print("All asynchronous invocations have been initialized.")
    sleep(860)
    print("Complete!")
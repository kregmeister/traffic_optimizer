import boto3
from botocore.config import Config
import psycopg2
import os
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from time import sleep

def delete_old_data(sql_connection, sql_cursor):
    # Since the entire process is invoked by a Step Function, old iterations' data is deleted before new data is generated
    sql_cursor.execute("""SELECT table_name
                FROM information_schema.tables
                WHERE table_schema='public'
                AND table_type='BASE TABLE';""")
    tables = sql_cursor.fetchall()
    
    count = 0 # Deletes tables in batches of 48 so RDS instance does not run out of memeory
    for table in tables:
        name = table[0]
        sql_cursor.execute(f"DROP TABLE IF EXISTS {name};")
        count += 1
        if count == 48:
            sql_connection.commit()
            count = 0 # Resets to 0 after 48 table delete commits accumulated
        
    sql_connection.commit()

# The entire month of June's traffic data cannot be generated in 15 minutes, so this parent divides it into independent, asynchronous parts
def lambda_handler(event, context):
    # Sets environment variables (for legibility)
    endpoint = os.environ.get('POSTGRES_ENDPOINT')
    pw = os.environ.get('POSTGRES_MASTER_PW')
    
    # Connects to RDS using password auth
    connection = psycopg2.connect(f"dbname=postgres user=postgres password={pw} host={endpoint} port=5432")
    cursor = connection.cursor()
    
    # Deletes all old data generated from other iterations of the program
    delete_old_data(connection, cursor)
    print("Old data deleted (if applicable).")
    
    connection.close()
    
    print("starting async invocation...")
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
    print(f" Lambda client initialized at {datetime.now()}")

    # Function to invoke Lambda asynchronously
    def invoke_lambda(date):
        payload = json.dumps({'date': date})
        response = client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:992382556703:function:traffic_data_generation',
            InvocationType='Event',  # Asynchronous invocation
            Payload=payload
        )
        return json.load(response['Payload'])

    dates = [datetime(2024, 6, 1) + timedelta(days=i) for i in range(30)] # Starts each day's data generation at 6/xx/2024 at 12:00AM
    
    # Use ThreadPoolExecutor to manage concurrent invocations
    with ThreadPoolExecutor(max_workers=15) as executor:
        # Map the function over the list of dates
        futures = [executor.submit(invoke_lambda, str(date)) for date in dates]

    # Continue processing after all invocations have completed
    print("All asynchronous invocations have been initialized.")
    sleep(600)
    return {"status": "Done"}
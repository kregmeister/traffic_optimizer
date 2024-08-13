import os
import psycopg2
from datetime import datetime
import boto3

def chart_construction(sorted_times, accidents_lst, total_drivers_sums, total_speeders_sums):
    # Sets MPLCONFIGDIR so Matplotlib can perform better
    os.environ['MPLCONFIGDIR'] = '/tmp'
    import matplotlib.pyplot as plt
    
    dates = [dt.date() for dt in accidents_lst]
    times = [dt.time() for dt in accidents_lst]
    
    # Converts times of day to strings so that they can be used in charts' x-axis ticks
    time_strs = [str(time) for time in sorted_times]
    
    # Craft histogram
    plt.figure(figsize=(11, 8))
    plt.hist([time.hour + time.minute/60 for time in times], bins=24, range=(0, 24), alpha=0.75, color='blue', ec='black')
    plt.title("Distribution of Accidents Across Hours of the Day")
    plt.xticks(range(len(time_strs[::4])), time_strs[::4], rotation=90)
    plt.xlabel("Hour of Day (Military)")
    plt.ylabel("Number of Accidents")
    plt.show()
    
    # Temporarily saves fig to Lambda environment, then uploads them to S3
    temp_accidents_hist = '/tmp/accidents_histogram.png'
    plt.savefig(temp_accidents_hist)
    plt.close()
    
    # Set up and save bar chart to S3 that shows percentages of speeders by each 15 minute interval
    plt.figure(figsize=(8, 11))
    speeding_rates_tod = [(s / tv) * 100 for tv, s in zip(total_drivers_sums, total_speeders_sums)]
    
    # Construct chart
    plt.bar(time_strs, speeding_rates_tod, align='edge')
    plt.title("Percentage of Vehicles That Sped on Highway X in June by Time of Day")
    plt.xlabel("Time of day (Military)")
    plt.ylabel("Ratio of Speeders (%)")
    plt.xticks(time_strs[::4], rotation=90)
    
    # Temporarily saves fig to Lambda environment, then uploads them to S3
    temp_speeding_chart = '/tmp/speeding_bar_chart.png'
    plt.savefig(temp_speeding_chart)
    plt.close()
    
    print("Charts saved to /tmp. Attempting to upload them to S3...")
    
    # Connect and upload chart to S3 bucket
    s3 = boto3.client('s3')
    
    # Saves speeding bar chart to S3 then deletes from Lambda environment
    with open(temp_speeding_chart, 'rb') as bar_chart:
        s3.upload_fileobj(bar_chart, "cjytc.other", "traffic_project/plots/pc_of_speeders_by_time_interval.png")
    os.remove(temp_speeding_chart)
    
    # Saves accidents scatter plot to S3 then deletes from Lambda environment
    with open(temp_accidents_hist, 'rb') as histogram:
        s3.upload_fileobj(histogram, "cjytc.other", "traffic_project/plots/june_accidents_histogram.png")
    os.remove(temp_accidents_hist)

def data_exploration(connection, cursor):
    # Gets all table names (2880 of them)
    cursor.execute("""SELECT table_name
                FROM information_schema.tables
                WHERE table_schema='public'
                AND table_type='BASE TABLE'
                ORDER BY table_name;""") # Preserves order of time (6/1 12:15 AM, 6/1 12:30 AM ...)
    tables = cursor.fetchall()
    tables = [t[0] for t in tables]
    print(f"Number of tables in DB: {len(tables)}")
    
    # Iterates through all tables so data can be described
    total_drivers = {}
    speeders = {}
    accidents_lst = []
    avg_speeds_weights = []
    for table in tables:
        # Convert table name (modified timestamp) into a datetime object
        dt_string = table.replace("dt_", "").replace("_", "-", 2).replace("_", " ", 1).replace("_", ":")
        dt = datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
        
        # Gets the volume of vehicles for the time interval
        cursor.execute(f"""SELECT COUNT(vehicle_type) FROM {table};""")
        volume = cursor.fetchall()[0][0]
        if dt.time() not in total_drivers.keys():
            total_drivers[dt.time()] = [volume]
        else:
            total_drivers[dt.time()].append(volume)
        
        # Number of speeders (Defined as driving 15 MPH over the limit)
        cursor.execute(f"""SELECT COUNT(velocity) FROM {table} WHERE velocity >= 70;""")
        speeding_vehicles = cursor.fetchall()[0][0]
        if dt.time() not in speeders.keys():
            speeders[dt.time()] = [speeding_vehicles]
        else:
            speeders[dt.time()].append(speeding_vehicles)
        
        # When accidents occurred (time of day)
        cursor.execute(f"""SELECT COUNT(accident) FROM {table} WHERE accident = 1;""")
        accidents = cursor.fetchall()[0][0]
        if accidents != 0:
            accidents_lst.append(dt)
        
        # Gets total average velocity and length of table so weighted average can later be calculated
        cursor.execute(f"""SELECT AVG(velocity), COUNT(*) FROM {table};""")
        avg_speed = cursor.fetchall()[0]
        avg_speeds_weights.append(avg_speed)
    
    # Sums of each time of days key (I.e. 6000 vehicles from 1:30 to 1:45 each day in June)
    total_drivers_sums = [sum(total_drivers[tod]) for tod in total_drivers.keys()]
    total_speeders_sums = [sum(speeders[tod]) for tod in speeders.keys()]
        
    # Accident rate (Accidents per million miles driven)
    total_accidents = len(accidents_lst)
    accident_rate = total_accidents / (sum(total_drivers_sums) * 5 / 1000000)
    print(f"Accidents per million miles driven: {accident_rate}")
    
    # Speeding rate (percent)
    speeding_rate = (sum(total_speeders_sums) / sum(total_drivers_sums)) * 100
    print(f"Percentage of drivers that were speeding: {speeding_rate}")
    
    # Average estimated time elapsed for the 5 miles (NoneTypes filtered out of avg_speeds)
    table_lens = [count for avg, count in avg_speeds_weights if avg != None]
    multiples = [avg * count for avg, count in avg_speeds_weights if avg != None]
    # Weighted average of velocity
    weighted_avg_speed = sum(multiples) / sum(table_lens)
    avg_time = (60 / weighted_avg_speed) * 5
    print(f"Average estimated time taken to travel the 5 mile stretch: {avg_time}")
    
    # Prints all accidents during the month and graphs when they occurred
    print(f"All accidents: {accidents_lst}")
    
    chart_construction(total_drivers.keys(), accidents_lst, total_drivers_sums, total_speeders_sums) # Since the tables are processed in order of time, the keys of total drivers will go from 12:00 AM to 11:45 PM
    
def lambda_handler(event, context):
    # Connects to RDS PostgreSQL DB
    conn = psycopg2.connect(f"dbname=postgres user=postgres password={os.environ.get("POSTGRES_PW")} host={os.environ.get("POSTGRES_HOST")}, port=5432")
    cur = conn.cursor()
    
    data_exploration(conn, cur)
    
    conn.commit()
    conn.close()
    print("Complete!")

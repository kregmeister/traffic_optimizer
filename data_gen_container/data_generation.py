#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:07:31 2024

@author: cjymain
"""

import numpy as np
from datetime import datetime, timedelta, time
import psycopg2
import os    

def generate_traffic_data(date: datetime, sql_connection, sql_cursor, number_of_intervals=96):
    # The distributions of velocity by vehicle type with first list item being mean velocity and the second being standard deviation 
    velocity_distributions = {"car": [61, 11], "suv": [58, 13], "truck": [55, 12], "18_wheeler": [51, 6], "service_vans": [53, 13]}
    
    # Adjusts values and variabilities for generated data based on time of week and time of day
    time_scales = {
            "weekday": 
            {
                "late_night": [0.925, (time(0, 0), time(5, 0)), [0.13, 0.11, 0.14, 0.56, 0.06], 18], # First list item is the velocity multiplier
                "normal_early_morn": [1, (time(5, 1), time(6, 59)), [0.14, 0.16, 0.20, 0.27, 0.23], 51], # Second list item is time range
                "rush_hour_am": [0.70, (time(7, 0), time(9, 30)), [0.24, 0.23, 0.18, 0.11, 0.24], 89], # Third list item is vehicle type probabilities
                "normal_midday": [0.975, (time(9, 31), time(16, 29)), [0.31, 0.28, 0.14, 0.09, 0.18], 71], # Fouth list item is car throughput per vehicle
                "rush_hour_pm": [0.65, (time(16, 30), time(19, 0)), [0.23, 0.22, 0.18, 0.10, 0.27], 86],
                "normal_evening": [1.00, (time(19, 1), time(21, 59)), [0.31, 0.30, 0.24, 0.07, 0.08], 55],
                "later_evening": [1.10, (time(22, 0), time(23, 59)), [0.33, 0.31, 0.21, 0.10, 0.05], 27]
                },
            "weekend": 
            {
                "late_night": [1.2, (time(0, 0), time(4, 0)), [0.47, 0.24, 0.14, 0.12, 0.03], 25],
                "quiet_morning": [0.90, (time(4, 1), time(8, 0)), [0.27, 0.24, 0.21, 0.16, 0.12], 31],
                "brunch_hours": [0.85, (time(8, 1), time(13, 0)), [0.31, 0.32, 0.24, 0.07, 0.06], 64],
                "afternoon": [1, (time(13, 1), time(20, 0)), [0.32, 0.30, 0.25, 0.05, 0.08], 66],
                "evening": [1.05, (time(20, 1), time(23, 59)), [0.36, 0.30, 0.22, 0.10, 0.02], 42]
                }
            }
    
    # Matches 15 minute time segment to timeframe it resides in
    def get_time_category(row_time, time_of_week):
        for period, details in time_scales[time_of_week].items():
            if details[1][0] <= row_time.time() <= details[1][1]:
                return period
        return None
    
    # Randomly chooses vehicle type based on timeframe's probabilities
    def choose_vehicle_type(probabilities):
        vehicle_types = ["car", "suv", "truck", "18_wheeler", "service_vans"]
        return np.random.choice(vehicle_types, p=probabilities)
    
    # Tracks whether an accident is active and its duration
    accident_duration_counter = 0
    accident_info = None

    for i in range(number_of_intervals):
        timestamp = date + timedelta(minutes=i*15)
        # Returns matching timeframe for current 15 minute window
        if timestamp.weekday() in range(5):
            period = get_time_category(timestamp, "weekday")
            velocity_multiplier, timeframe, vtype_probabilities, volume_per_minute = time_scales["weekday"][period]
        else:
            period = get_time_category(timestamp, "weekend")
            velocity_multiplier, timeframe, vtype_probabilities, volume_per_minute = time_scales["weekend"][period]
            
        # Format timestamp to be able to be table name (can't start with digit, no spaces, etc.)
        timestamp = "dt_" + str(timestamp).replace("-", "_").replace(" ", "_").replace(":", "_")
        
        # Since volume is in cars/minute, its randomized for variance then multiplied by 15 to give total volume for 15 minute window
        volume = int(np.random.normal(loc=volume_per_minute, scale=5)) * 15
        
        if accident_info != None: # Means an accident is active
            accident_duration_counter += 1
            if accident_duration_counter > accident_info['duration']: # Sets accident bool to inactive if the chosen duration of accident has passed
                accident_info = None
        else:
            accident_info = None
        
        # Creates new table for each 15 minute timeframe
        sql_cursor.execute(f"""CREATE TABLE {timestamp} (
                                vehicle_type TEXT NOT NULL,
                                velocity FLOAT NOT NULL,
                                accident INT NOT NULL
                            );""")
        
        def sim_15_min(num_vehicles, accident_info):
            # Applies a multiplier that slows velocities and throughput due to the active accident   
            if accident_info != None:
                accident_multiplier = accident_info['lanes_open'] / 3 # Slows down speeds by how many lanes are still open (I.E. 2 lanes = 66% speeds)
                if accident_multiplier == 0:
                    accident_multiplier = 0.25 / 3 # So velocities never equal 0
            else:
                accident_multiplier = 1
            
            # Accident multiplier is applied to how many vehicles passed the sensor during the 15 minutes because, inevitably, less cars will get through during a traffic jam
            num_vehicles = int(num_vehicles * accident_multiplier)
            
            if num_vehicles > 750:
                volume_multiplier = 750 / num_vehicles # More than 750 vehicles in the 15 minutes means its crowded enough to slow speeds down, proportional to 
            else:
                volume_multiplier = 1
            
            # Simulate data for each vehicle in the time period
            for vehicle in range(num_vehicles):
                # Sets the velocity multiplier from the time period with a bit of variance
                time_of_day_multiplier = np.random.normal(loc=velocity_multiplier, scale=0.05)
                
                # Concatenates all multipliers
                full_multiplier = time_of_day_multiplier * accident_multiplier * volume_multiplier
                
                # Randomly chooses vehicle type based on probabilities of time period
                vtype = choose_vehicle_type(vtype_probabilities)
                
                # Chooses correct avg speeds and stdevs based on vehicle type
                velocity_base, velocity_std = velocity_distributions[vtype]
                
                # Will very occasionally choose a negative velocity, which in this scenario is not possible
                valid = False
                while valid == False:
                    speed = np.random.normal(loc=velocity_base, scale=velocity_std) * full_multiplier
                    if speed > 0:
                        valid = True
                        
                # Uses random chance to determine if there is an accident. Then, an accident's severity (duration and lane closings) is also randomly determined
                crash = [True, False]
                if accident_info != None:
                    accident = False
                elif speed <= 30:
                    chance = 1.00 / 200000 # The FHWA generally cites an accident rate of 1-4 accidents per million vehicle miles traveled, so: accident_rate = estimated_accidents/million / (million_miles / miles_traveled_per_car)
                    accident = np.random.choice(crash, p=[chance, 1-chance])
                elif 30 < speed <= 45:
                    chance = 1.25 / 200000
                    accident = np.random.choice(crash, p=[chance, 1-chance])
                elif 45 < speed <= 55:
                    chance = 1.75 / 200000
                    accident = np.random.choice(crash, p=[chance, 1-chance])
                elif 55 < speed <= 65:
                    chance = 2.75 / 200000
                    accident = np.random.choice(crash, p=[chance, 1-chance])
                elif 65 < speed <= 75:
                    chance = 5.25 / 200000
                    accident = np.random.choice(crash, p=[chance, 1-chance])
                elif 75 <= speed:
                    chance = 15.00 / 200000
                    accident = np.random.choice(crash, p=[chance, 1-chance])
                
                if accident == True:
                    lanes_open = [2, 1, 0] # Number of lanes still in operation
                    duration = [30, 60, 90, 120, 150] # How long until lanes are re-opened (in minutes)
                    
                    accident_info = {}
                    accident_info['lanes_open'] = np.random.choice(lanes_open)
                    accident_info['duration'] = np.random.choice(duration) / 15 # Number of 15 minute periods accident will span
                
                accident_bool = int(accident) # Accident occurrence is 1, no accident is 0
                
                # Appends row to time period's table
                sql_cursor.execute(f"""INSERT INTO {timestamp} (vehicle_type, velocity, accident) VALUES ('{vtype}', {speed}, {accident_bool})""")

            sql_connection.commit()
            return accident_info
        
        accident_info = sim_15_min(volume, accident_info)



def lambda_handler(event, context):
    # Connects to RDS
    connection = psycopg2.connect(f"dbname=postgres user=postgres password={os.environ.get('POSTGRES_MASTER_PW')} host={os.environ.get('POSTGRES_ENDPOINT')} port=5432")
    cursor = connection.cursor()
    
    date_str = event.get('date')
    date = datetime.fromisoformat(date_str)
    
    # Calls data generation function from above, passing cursor
    generate_traffic_data(date, connection, cursor)
    connection.close()
    
    return {
        'statusCode': 200,
        'body': 'Traffic data for June 2024 generated and written to PostgreSQL RDS'
    }





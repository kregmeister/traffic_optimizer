#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:54:02 2024

@author: cjymain
"""

import json
import os
import s3fs
import glob
import numpy as np

def evaluate_forecasts(cwd, fs):
    # Sets MPLCONFIGDIR so Matplotlib can perform better
    os.environ['MPLCONFIGDIR'] = '/tmp'
    import matplotlib.pyplot as plt

    def cluster_summary_chart(solution, color):
	    
        if solution == "original":
            return
		# Original is always included for comparison
        files = fs.glob(cwd + "/model_evals/original/cluster_summary.json/*.json")
		
        path = cwd + f"/model_evals/{solution}/cluster_summary.json/*.json"
        files.append(fs.glob(path)[0])
		
		# Sets blue as original's color and color from dict as solution's color
        colors = ["b", color]
        labels = ["original", solution]
		
        plt.figure(figsize=(12, 8))
        for i in range(len(files)):
            combined_summary = []
            with fs.open(files[i], "r") as f:
                for line in f:
                    combined_summary.append(json.loads(line))
				
			# Extract data for plotting
            cluster_num = [stat['cluster_num'] for stat in combined_summary]
            means_f1 = [stat['mean_vel'] for stat in combined_summary]
            means_f2 = [stat['mean_time'] for stat in combined_summary]
            stddevs_f1 = [stat['stddev_vel'] for stat in combined_summary]
            stddevs_f2 = [stat['stddev_time'] for stat in combined_summary]
			
            plt.errorbar(means_f1, means_f2, xerr=stddevs_f1, yerr=stddevs_f2, fmt='o', capsize=5, label=f'{labels[i]}: Cluster Centers (±1 SD)', color=colors[i]) # Shows the shape of each cluster
			
            for i in range(len(combined_summary)): # Labels each point to show which cluster is which
                plt.text(means_f1[i] + 1.5, means_f2[i] + 2.25, f"Cluster {combined_summary[i]['cluster_num']}")
        plt.axvline(x=55, color='red', linestyle='--', linewidth=2, label="Speed Limit (55 MPH)")
        plt.title('Cluster Centers with Standard Deviation')
        plt.xlabel('Velocity (MPH)')
        plt.ylabel('Travel Time (Minutes/5 Miles)')
        plt.grid(True)
        plt.legend()
		
		# Uploads plot to S3 as a .PNG
        with fs.open(cwd + f"/plots/cluster_plot_{solution}.png", 'wb') as plot:
            plt.savefig(plot, format='png')
        plt.close()
	
    def forecast_results(solution):
        c_path = cwd + "/model_evals/original/cluster_summary.json/*.json"
        a_path = cwd + "/model_evals/original/accidents_details.json/*.json"

        dicts = []
        for path in [c_path, a_path]:
            data = []

            files = fs.glob(path)

            with fs.open(files[0], "r") as f:
                for line in f:
                    data.append(json.loads(line))
            subdict = {k: [json[k] for json in data] for k in data[0]}
            dicts.append(subdict)

        cluster_df = dicts[0]
        accidents_df = dicts[1]

        total_accidents = len(accidents_df["datetime"])
        total_vehicles = sum(cluster_df["count"])
        total_speeders = sum(cluster_df["speeders_count"])

        accident_rate = total_accidents / (total_vehicles * 5 / 1000000)

        percent_speeders = (total_speeders / total_vehicles) * 100

        mean_times_count = [mean * count for mean, count in zip(cluster_df["mean_time"], cluster_df["count"])]
        average_time = sum(mean_times_count) / sum(cluster_df["count"])
		
        return accident_rate, percent_speeders, average_time
	
    solutions_dict = {"original": "b", "no_solution": "g", "increase_police_presence": "r", "plus_5_speed_limit": "purple", "minus_5_speed_limit": "y", "extra_lane": "orange"}
	
    accident_rates = []
    percent_speeders = []
    average_times = []
    for solution, color in solutions_dict.items():
        cluster_summary_chart(solution, color)
		
        ar, ps, at = forecast_results(solution)
        accident_rates.append(ar)
        percent_speeders.append(ps)
        average_times.append(at)
	
    x = np.arange(len(solutions_dict.keys()))  # the label locations
    width = 0.2  # the width of the bars
	
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, accident_rates, width, label='Accidents per Million Miles Driven')
    rects2 = ax.bar(x, percent_speeders, width, label='Percent of Vehicles Speeding')
    rects3 = ax.bar(x + width, average_times, width, label='Average Minutes to Travel 5 Miles')
	
	# Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Solutions')
    ax.set_ylabel('Values')
    ax.set_title('Traffic Forecast Summary Statistics by Solution')
    ax.set_xticks(x)
    ax.set_xticklabels(solutions_dict.keys())
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend()
	
	# Uploads plot to S3 as a .PNG
    with fs.open(cwd + "/plots/forecast_summaries_plot.png", 'wb') as plot:
        plt.savefig(plot, format='png')
    plt.close()

def lambda_handler(event, context):
    # Create s3 file system
    fs = s3fs.S3FileSystem()
    
    # Fetches model summary JSON
    cwd = "s3://cjytc.other/traffic_project/"
    
    evaluate_forecasts(cwd, fs)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }



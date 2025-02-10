import matplotlib.pyplot as plt
import pandas as pd

# Data setup
data_dvd = {
    'Number of Nodes': [20, 30, 40, 50, 60],
    'DVD Latency (seconds)': [0.66, 0.63, 0.62, 0.61, 0.59],
    'DVD Throughput (objects/sec)': [863.64, 904.76, 919.35, 934.43, 966.10]
}
data_faiss = {
    'Number of Nodes': [20, 30, 40, 50, 60],
    'FAISS Latency (seconds)': [0.4222, 0.3402, 0.2648, 0.2107, 0.1379],
    'FAISS Throughput (objects/sec)': [1347.79, 1672.68, 2148.81, 2700.79, 4126.12]
}
data_waieate = {
    'Number of Nodes': [20, 30, 40, 50, 60],
    'Waieate Latency (seconds)': [0.4722, 0.3702, 0.2748, 0.2707, 0.1779],
    'Waieate Throughput (objects/sec)': [1327.79, 1622.68, 2128.81, 2680.79, 4116.12]
}

df_dvd = pd.DataFrame(data_dvd)
df_faiss = pd.DataFrame(data_faiss)
df_waieate = pd.DataFrame(data_waieate)

# Calculate latency overhead percentage
overhead_latency_faiss_percent = abs(100 * (df_dvd['DVD Latency (seconds)'] - df_faiss['FAISS Latency (seconds)']) / df_faiss['FAISS Latency (seconds)'])
overhead_latency_waieate_percent = abs(100 * (df_dvd['DVD Latency (seconds)'] - df_waieate['Waieate Latency (seconds)']) / df_waieate['Waieate Latency (seconds)'])

# Calculate throughput overhead percentage correctly
overhead_throughput_faiss_percent = abs(100 * (df_dvd['DVD Throughput (objects/sec)'] - df_faiss['FAISS Throughput (objects/sec)']) / df_faiss['FAISS Throughput (objects/sec)'])
overhead_throughput_waieate_percent = abs(100 * (df_dvd['DVD Throughput (objects/sec)'] - df_waieate['Waieate Throughput (objects/sec)']) / df_waieate['Waieate Throughput (objects/sec)'])

# Plotting Latency with grouped bars and overhead percentages
fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()
bar_width = 1  # Adjust bar width for better visibility

ax1.bar(df_dvd['Number of Nodes'] - bar_width, df_dvd['DVD Latency (seconds)'], color='blue', width=bar_width, label='DVD Latency')
ax1.bar(df_dvd['Number of Nodes'], df_faiss['FAISS Latency (seconds)'], color='orange', width=bar_width, label='FAISS Latency')
ax1.bar(df_dvd['Number of Nodes'] + bar_width, df_waieate['Waieate Latency (seconds)'], color='green', width=bar_width, label='Waieate Latency')

ax2.plot(df_dvd['Number of Nodes'], overhead_latency_faiss_percent, 'r--', marker='o', linewidth=2, label='Overhead vs FAISS (%)')
ax2.plot(df_dvd['Number of Nodes'], overhead_latency_waieate_percent, 'm--', marker='o', linewidth=2, label='Overhead vs Waieate (%)')

ax1.set_xlabel('Number of Nodes')
ax1.set_ylabel('Latency (seconds)')
ax2.set_ylabel('Overhead (%)')
ax1.legend(loc='upper left', bbox_to_anchor=(0.07, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(0.24, 1))
plt.show()

# Plotting Throughput with bars and overhead percentages as lines
fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()

ax1.bar(df_dvd['Number of Nodes'] - bar_width, df_dvd['DVD Throughput (objects/sec)'], color='blue', width=bar_width, label='DVD Throughput')
ax1.bar(df_dvd['Number of Nodes'], df_faiss['FAISS Throughput (objects/sec)'], color='orange', width=bar_width, label='FAISS Throughput')
ax1.bar(df_dvd['Number of Nodes'] + bar_width, df_waieate['Waieate Throughput (objects/sec)'], color='green', width=bar_width, label='Waieate Throughput')

ax2.plot(df_dvd['Number of Nodes'], overhead_throughput_faiss_percent, 'r--', marker='o', linewidth=2, label='Throughput Overhead vs FAISS (%)')
ax2.plot(df_dvd['Number of Nodes'], overhead_throughput_waieate_percent, 'm--', marker='o', linewidth=2, label='Throughput Overhead vs Waieate (%)')

ax1.set_xlabel('Number of Nodes')
ax1.set_ylabel('Throughput (objects/sec)')
ax2.set_ylabel('Overhead (%)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper center', bbox_to_anchor=(0.35, 1))
plt.show()

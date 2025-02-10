import matplotlib.pyplot as plt
import pandas as pd

# Data setup
data_dvd = {
    'Number of Objects': [114, 228, 342, 456, 570],
    'DVD Latency (seconds)': [0.49, 0.51, 0.53, 0.57, 0.59],
    'DVD Throughput (objects/sec)': [247.83, 475, 645.28, 800, 966.10]
}
data_faiss = {
    'Number of Objects': [114, 228, 342, 456, 570],
    'FAISS Latency (seconds)': [0.3988, 0.4082, 0.3980, 0.4040, 0.4222],
    'FAISS Throughput (objects/sec)': [285.85, 558.57, 859.29, 1128.80, 1347.79]
}
data_waieate = {
    'Number of Objects': [114, 228, 342, 456, 570],
    'Waieate Latency (seconds)': [0.4988, 0.5082, 0.4980, 0.5040, 0.5222],
    'Waieate Throughput (objects/sec)': [228.54, 448.57, 686.29, 904.80, 1091.79]
}

df_dvd = pd.DataFrame(data_dvd)
df_faiss = pd.DataFrame(data_faiss)
df_waieate = pd.DataFrame(data_waieate)

# Calculate latency overhead percentage
overhead_latency_faiss_percent = 100 * abs(df_dvd['DVD Latency (seconds)'] - df_faiss['FAISS Latency (seconds)']) / df_faiss['FAISS Latency (seconds)']
overhead_latency_waieate_percent = 100 * abs(df_dvd['DVD Latency (seconds)'] - df_waieate['Waieate Latency (seconds)']) / df_waieate['Waieate Latency (seconds)']

# Plotting the latency without log scale
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Number of Objects')
ax1.set_ylabel('Latency (seconds)', color='tab:blue')
ax1.bar(df_dvd['Number of Objects'] - 15, df_dvd['DVD Latency (seconds)'], width=15, color='b', label='DVD')
ax1.bar(df_faiss['Number of Objects'], df_faiss['FAISS Latency (seconds)'], width=15, color='orange', label='FAISS')
ax1.bar(df_waieate['Number of Objects'] + 15, df_waieate['Waieate Latency (seconds)'], width=15, color='g', label='Waieate')
ax1.set_xticks(df_dvd['Number of Objects'])
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel('Latency Overhead (%)', color='tab:red')
ax2.plot(df_dvd['Number of Objects'], overhead_latency_faiss_percent, 'r--', label='Overhead vs FAISS', color='red')
ax2.plot(df_dvd['Number of Objects'], overhead_latency_waieate_percent, 'm--', label='Overhead vs Waieate', color='magenta')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper center', bbox_to_anchor=(0.22, 1))

fig.tight_layout()
plt.show()

# Calculate throughput overhead percentage
overhead_throughput_faiss_percent = 100 * abs(df_faiss['FAISS Throughput (objects/sec)'] - df_dvd['DVD Throughput (objects/sec)']) / df_faiss['FAISS Throughput (objects/sec)']
overhead_throughput_waieate_percent = 100 * abs(df_waieate['Waieate Throughput (objects/sec)'] - df_dvd['DVD Throughput (objects/sec)']) / df_waieate['Waieate Throughput (objects/sec)']

# Plotting throughput as bars
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Number of Objects')
ax1.set_ylabel('Throughput (objects/sec)', color='tab:blue')
ax1.bar(df_dvd['Number of Objects'] - 15, df_dvd['DVD Throughput (objects/sec)'], width=15, color='blue', label='DVD Throughput')
ax1.bar(df_faiss['Number of Objects'], df_faiss['FAISS Throughput (objects/sec)'], width=15, color='orange', label='FAISS Throughput')
ax1.bar(df_waieate['Number of Objects'] + 15, df_waieate['Waieate Throughput (objects/sec)'], width=15, color='green', label='Waieate Throughput')
ax1.set_xticks(df_dvd['Number of Objects'])
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel('Throughput Overhead (%)', color='tab:red')
ax2.plot(df_dvd['Number of Objects'], overhead_throughput_faiss_percent, 'r--', label='Throughput Overhead vs FAISS', color='red')
ax2.plot(df_dvd['Number of Objects'], overhead_throughput_waieate_percent, 'm--', label='Throughput Overhead vs Waieate', color='magenta')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper center', bbox_to_anchor=(0.38, 1))

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('pid_log_20250918_205041.csv')  # change filename
plt.figure()
plt.plot(df['time_s'], df['dist'], label='Dist (m)')
plt.plot(df['time_s'], df['error'], label='Error (m)')
plt.plot(df['time_s'], df['velocity'], label='Velocity (m/s)')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Time [s]')
plt.legend()
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Rocket parameters (from reference scripts)
gravity = 9.81  # m/s²
air_density = 1.225  # kg/m³ at sea level
drag_coefficient = 0.5179
cross_sectional_area = 0.01  # m²
rocket_mass = 27.021  # Dry mass in kg (after burn)
drogue_velocity = -27.79  # m/s (descent after drogue)
main_velocity = -8.97  # m/s (descent after main)
main_deploy_altitude = 457.2  # meters
total_flight_time = 362.0  # seconds
burn_phase_duration = 4.517  # seconds (from reference scripts)

# Read CSV file
df = pd.read_csv('rocket_acceleration_data_dummy.csv')
time = df['timestamp'].values
accel_x_g = df['accel_x_g'].values
accel_y_g = df['accel_y_g'].values
accel_z_g = df['accel_z_g'].values

# Debug: Print max accel_z_g to check data
print(f"Max accel_z_g: {max(accel_z_g):.2f} G")
print(f"Min accel_z_g: {min(accel_z_g):.2f} G")

# Convert accelerations from G to m/s²
accel_x = accel_x_g * gravity
accel_y = accel_y_g * gravity
accel_z = accel_z_g * gravity

# Initialize lists for velocity and distance
vel_x = [0.0]
vel_y = [0.0]
vel_z = [0.0]
dist_x = [0.0]
dist_y = [0.0]
dist_z = [0.0]

# Step 1: Numerical integration until end of burn phase or apogee
apogee_idx = None
for i in range(1, len(time)):
    if time[i] > 50:  # Limit initial integration
        break
    dt = time[i] - time[i-1]
    # Apply drag only after burn phase
    if time[i] > burn_phase_duration and vel_z[-1] > 0:
        drag_force = 0.5 * air_density * drag_coefficient * cross_sectional_area * (vel_z[-1] ** 2)
        net_accel_z = accel_z[i] - (drag_force / rocket_mass)
    else:
        net_accel_z = accel_z[i]  # Use CSV acceleration during burn
    # Integrate accelerations to velocities
    vel_x.append(vel_x[-1] + accel_x[i] * dt)
    vel_y.append(vel_y[-1] + accel_y[i] * dt)
    vel_z.append(vel_z[-1] + net_accel_z * dt)
    # Integrate velocities to distances
    dist_x.append(dist_x[-1] + vel_x[-1] * dt)
    dist_y.append(dist_y[-1] + vel_y[-1] * dt)
    dist_z.append(max(0, dist_z[-1] + vel_z[-1] * dt))
    # Detect apogee after burn phase
    if time[i] > burn_phase_duration and vel_z[-1] <= 0 and apogee_idx is None:
        apogee_idx = i
        apogee_time = time[i]
        apogee_dist = dist_z[-1]
        print(f"Apogee detected at t = {apogee_time:.2f} s, vel_z = {vel_z[-1]:.2f} m/s, dist_z = {dist_z[-1]:.2f} m")

# Step 2: Extend to apogee if not reached
if apogee_idx is None:
    current_time = time[-1]
    current_vel_z = vel_z[-1]
    current_dist_z = dist_z[-1]
    dt = 0.1
    while current_vel_z > 0:
        current_time += dt
        drag_force = 0.5 * air_density * drag_coefficient * cross_sectional_area * (current_vel_z ** 2)
        net_accel_z = -gravity - (drag_force / rocket_mass)  # Gravity and drag only
        current_vel_z += net_accel_z * dt
        current_dist_z += current_vel_z * dt
        time = np.append(time, current_time)
        accel_x = np.append(accel_x, 0)
        accel_y = np.append(accel_y, 0)
        accel_z = np.append(accel_z, net_accel_z)
        accel_x_g = np.append(accel_x_g, 0)
        accel_y_g = np.append(accel_y_g, 0)
        accel_z_g = np.append(accel_z_g, net_accel_z / gravity)
        vel_x = np.append(vel_x, vel_x[-1])
        vel_y = np.append(vel_y, vel_y[-1])
        vel_z = np.append(vel_z, current_vel_z)
        dist_x = np.append(dist_x, dist_x[-1])
        dist_y = np.append(dist_y, dist_y[-1])
        dist_z = np.append(dist_z, max(0, current_dist_z))
    apogee_idx = len(time) - 1
    apogee_time = current_time
    apogee_dist = current_dist_z
    print(f"Apogee extended at t = {apogee_time:.2f} s, dist_z = {apogee_dist:.2f} m")

# Step 3: Drogue phase
current_time = apogee_time
current_vel_z = drogue_velocity
current_dist_z = apogee_dist
# Truncate arrays at apogee
time = time[:apogee_idx + 1]
accel_x = accel_x[:apogee_idx + 1]
accel_y = accel_y[:apogee_idx + 1]
accel_z = accel_z[:apogee_idx + 1]
accel_x_g = accel_x_g[:apogee_idx + 1]
accel_y_g = accel_y_g[:apogee_idx + 1]
accel_z_g = accel_z_g[:apogee_idx + 1]
vel_x = vel_x[:apogee_idx + 1]
vel_y = vel_y[:apogee_idx + 1]
vel_z = vel_z[:apogee_idx + 1]
dist_x = dist_x[:apogee_idx + 1]
dist_y = dist_y[:apogee_idx + 1]
dist_z = dist_z[:apogee_idx + 1]

dt = 0.1
while current_dist_z > main_deploy_altitude and current_dist_z > 0:
    current_time += dt
    accel_z_val = 0  # Constant velocity
    current_dist_z += current_vel_z * dt
    time = np.append(time, current_time)
    accel_x = np.append(accel_x, 0)
    accel_y = np.append(accel_y, 0)
    accel_z = np.append(accel_z, 0)
    accel_x_g = np.append(accel_x_g, 0)
    accel_y_g = np.append(accel_y_g, 0)
    accel_z_g = np.append(accel_z_g, 0)
    vel_x = np.append(vel_x, vel_x[-1])
    vel_y = np.append(vel_y, vel_y[-1])
    vel_z = np.append(vel_z, current_vel_z)
    dist_x = np.append(dist_x, dist_x[-1])
    dist_y = np.append(dist_y, dist_y[-1])
    dist_z = np.append(dist_z, max(0, current_dist_z))

# Step 4: Main parachute deployment
current_vel_z = main_velocity
while current_dist_z > 0:
    current_time += dt
    accel_z_val = 0  # Constant velocity
    current_dist_z += current_vel_z * dt
    time = np.append(time, current_time)
    accel_x = np.append(accel_x, 0)
    accel_y = np.append(accel_y, 0)
    accel_z = np.append(accel_z, 0)
    accel_x_g = np.append(accel_x_g, 0)
    accel_y_g = np.append(accel_y_g, 0)
    accel_z_g = np.append(accel_z_g, 0)
    vel_x = np.append(vel_x, vel_x[-1])
    vel_y = np.append(vel_y, vel_y[-1])
    vel_z = np.append(vel_z, current_vel_z)
    dist_x = np.append(dist_x, dist_x[-1])
    dist_y = np.append(dist_y, dist_y[-1])
    dist_z = np.append(dist_z, max(0, current_dist_z))

# Adjust time to match 362 seconds
if time[-1] != total_flight_time:
    time_scale = total_flight_time / time[-1]
    time = time * time_scale

# Truncate acceleration data for plotting (up to 45 seconds)
accel_truncate_idx = next((i for i, t in enumerate(time) if t > 45), len(time))
time_accel = time[:accel_truncate_idx]
accel_x_g_truncated = accel_x_g[:accel_truncate_idx]
accel_y_g_truncated = accel_y_g[:accel_truncate_idx]
accel_z_g_truncated = accel_z_g[:accel_truncate_idx]

# Plot acceleration (0–45 seconds, in G)
plt.figure(figsize=(10, 6))
plt.plot(time_accel, accel_x_g_truncated, 'r-', label='Acceleration X')
plt.plot(time_accel, accel_y_g_truncated, 'g-', label='Acceleration Y')
plt.plot(time_accel, accel_z_g_truncated, 'b-', label='Acceleration Z')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (G)')
plt.title('Acceleration vs Time')
plt.legend()
plt.grid(True)
plt.xlim(0, 45)
plt.savefig('acceleration_from_imu.png')

# Plot velocity (0–362 seconds)
plt.figure(figsize=(10, 6))
plt.plot(time, vel_x, 'r-', label='Velocity X')
plt.plot(time, vel_y, 'g-', label='Velocity Y')
plt.plot(time, vel_z, 'b-', label='Velocity Z')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time')
plt.legend()
plt.grid(True)
plt.xlim(0, 362)
plt.savefig('velocity_from_imu.png')

# Plot distance (0–362 seconds)
plt.figure(figsize=(10, 6))
plt.plot(time, dist_x, 'r-', label='Distance X')
plt.plot(time, dist_y, 'g-', label='Distance Y')
plt.plot(time, dist_z, 'b-', label='Distance Z')
plt.xlabel('Time (seconds)')
plt.ylabel('Distance (m)')
plt.title(f'Distance vs Time (Apogee: {max(dist_z):.2f} m)')
plt.legend()
plt.grid(True)
plt.xlim(0, 362)
plt.savefig('distance_from_imu.png')

print(f"Total flight time: {time[-1]:.2f} seconds")
print(f"Apogee reached at t = {apogee_time:.2f} s, altitude = {apogee_dist:.2f} m")
print(f"Max apogee reached: {max(dist_z):.2f} meters")
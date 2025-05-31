import matplotlib.pyplot as plt
import numpy as np

# Rocket parameters
initial_mass = 36.8  # Initial wet mass in kg
propellant_initial_mass = 9.779  # Initial propellant mass in kg
gravity = 9.81  # m/s²
total_flight_time = 362.0  # seconds
drogue_velocity = -27.79   # m/s (constant descent after drogue deployment)
main_velocity = -8.93  # m/s (constant descent after main deployment)
main_deploy_altitude = 457.2  # 1,500 ft converted to meters
target_apogee = 9144.0  # Target apogee in meters

# Drag parameters
air_density = 1.225  # kg/m³ at sea level
drag_coefficient = 0.5179 # Adjusted to aim for ~9144 m apogee
cross_sectional_area = 0.01  # m² (assuming 98 mm diameter, A = π*(0.049)^2)

# RSE data from the document (burn phase: 0 to 4.517 seconds)
time_data = [0.0, 0.02, 0.063, 0.116, 0.206, 0.349, 0.578, 0.921, 1.378, 1.834, 2.29, 2.746, 2.87, 3.202, 3.659, 3.999, 4.19, 4.38, 4.487, 4.517]
thrust_data = [0.0, 97.5153, 5878.54, 6146.99, 6321.0, 6313.79, 6286.77, 6281.54, 6265.53, 6213.6, 6137.53, 6164.89, 6079.0, 4268.06, 2843.96, 1793.96, 1242.27, 513.758, 77.8319, 0.0]
propellant_mass_data = [9779.0, 9778.57, 9722.03, 9581.81, 9334.93, 8937.42, 8302.58, 7354.13, 6092.58, 4840.62, 3601.49, 2367.26, 2033.23, 1277.44, 562.366, 215.434, 87.8456, 14.4404, 0.513715, 0.0]

# Convert propellant mass to kg
propellant_mass_data = [m / 1000.0 for m in propellant_mass_data]

# Lists to store the full flight data
time_full = time_data.copy()
accel_x_full = [0] * len(time_data)  # No thrust in x, y
accel_y_full = [0] * len(time_data)
accel_z_full = []
vel_x = [0]  # Initial velocities
vel_y = [0]
vel_z = [0]
dist_x = [0]  # Initial positions
dist_y = [0]
dist_z = [0]

# Step 1: Burn phase (0 to 4.517 seconds) with mass loss
for i in range(len(time_data)):
    current_mass = initial_mass - (initial_mass - 27.021) * (propellant_mass_data[i] / propellant_initial_mass)
    accel_z = (thrust_data[i] / current_mass) - gravity if current_mass > 0 else 0
    accel_z_full.append(accel_z)

for i in range(1, len(time_data)):
    dt = time_data[i] - time_data[i-1]
    vel_x.append(vel_x[-1] + accel_x_full[i] * dt)
    vel_y.append(vel_y[-1] + accel_y_full[i] * dt)
    vel_z.append(vel_z[-1] + accel_z_full[i] * dt)
    dist_x.append(dist_x[-1] + vel_x[-1] * dt)
    dist_y.append(dist_y[-1] + vel_y[-1] * dt)
    dist_z.append(dist_z[-1] + vel_z[-1] * dt)

# Step 2: Coasting phase (after burn, gravity and drag act until drogue deployment at 44.44 m/s upward)
dt = 0.1  # Time step
current_time = time_full[-1]
current_vel_z = vel_z[-1]
current_dist_z = dist_z[-1]
rocket_mass = 27.021  # Mass after propellant is consumed

while current_vel_z > 0 and current_time < 50:  # Limit to 50 seconds
    current_time += dt
    # Drag force: F_d = 0.5 * ρ * C_d * A * v^2 (opposes motion)
    drag_force = 0.5 * air_density * drag_coefficient * cross_sectional_area * (current_vel_z ** 2) if current_vel_z > 0 else 0
    net_force = -rocket_mass * gravity - drag_force
    accel_z = net_force / rocket_mass
    current_vel_z += accel_z * dt
    current_dist_z += current_vel_z * dt
    time_full.append(current_time)
    accel_x_full.append(0)
    accel_y_full.append(0)
    accel_z_full.append(accel_z)
    vel_x.append(vel_x[-1])
    vel_y.append(vel_y[-1])
    vel_z.append(current_vel_z)
    dist_x.append(dist_x[-1])
    dist_y.append(dist_y[-1])
    dist_z.append(current_dist_z)

# Truncate data to 50 seconds for acceleration graph
accel_truncate_idx = next((i for i, t in enumerate(time_full) if t > 50), len(time_full))
time_full_accel = time_full[:accel_truncate_idx]
accel_x_full_truncated = accel_x_full[:accel_truncate_idx]
accel_y_full_truncated = accel_y_full[:accel_truncate_idx]
accel_z_full_truncated = accel_z_full[:accel_truncate_idx]

# Continue simulation beyond 50 seconds for velocity and distance
while current_vel_z > 0:
    current_time += dt
    drag_force = 0.5 * air_density * drag_coefficient * cross_sectional_area * (current_vel_z ** 2) if current_vel_z > 0 else 0
    net_force = -rocket_mass * gravity - drag_force
    accel_z = net_force / rocket_mass
    current_vel_z += accel_z * dt
    current_dist_z += current_vel_z * dt
    time_full.append(current_time)
    accel_x_full.append(0)
    accel_y_full.append(0)
    accel_z_full.append(accel_z)
    vel_x.append(vel_x[-1])
    vel_y.append(vel_y[-1])
    vel_z.append(current_vel_z)
    dist_x.append(dist_x[-1])
    dist_y.append(dist_y[-1])
    dist_z.append(current_dist_z)

# Step 3: Drogue parachute deployment (velocity set to -44.44 m/s when upward velocity was 44.44 m/s)
drogue_deploy_time = None
for i in range(len(vel_z) - 1, -1, -1):
    if vel_z[i] >= 44.44:
        drogue_deploy_time = time_full[i]
        drogue_deploy_dist = dist_z[i]
        break
if drogue_deploy_time is not None:
    current_time = drogue_deploy_time
    current_vel_z = drogue_velocity  # Immediate change to constant descent
    current_dist_z = drogue_deploy_dist
    time_full = time_full[:i + 1]
    accel_x_full = accel_x_full[:i + 1]
    accel_y_full = accel_y_full[:i + 1]
    accel_z_full = accel_z_full[:i + 1]
    vel_x = vel_x[:i + 1]
    vel_y = vel_y[:i + 1]
    vel_z = vel_z[:i + 1]
    dist_x = dist_x[:i + 1]
    dist_y = dist_y[:i + 1]
    dist_z = dist_z[:i + 1]

while current_dist_z > main_deploy_altitude and current_dist_z > 0:
    current_time += dt
    accel_z = 0  # Constant velocity
    current_dist_z += current_vel_z * dt
    time_full.append(current_time)
    accel_x_full.append(0)
    accel_y_full.append(0)
    accel_z_full.append(accel_z)
    vel_x.append(vel_x[-1])
    vel_y.append(vel_y[-1])
    vel_z.append(current_vel_z)
    dist_x.append(dist_x[-1])
    dist_y.append(dist_y[-1])
    dist_z.append(current_dist_z)

# Step 4: Main parachute deployment (velocity set to -27.79 m/s)
current_vel_z = main_velocity  # Immediate change to constant descent
while current_dist_z > 0:
    current_time += dt
    accel_z = 0  # Constant velocity
    current_dist_z += current_vel_z * dt
    time_full.append(current_time)
    accel_x_full.append(0)
    accel_y_full.append(0)
    accel_z_full.append(accel_z)
    vel_x.append(vel_x[-1])
    vel_y.append(vel_y[-1])
    vel_z.append(current_vel_z)
    dist_x.append(dist_x[-1])
    dist_y.append(dist_y[-1])
    dist_z.append(current_dist_z)

# Adjust time to match total_flight_time
if len(time_full) > 2:
    time_scale = total_flight_time / time_full[-1]
    time_full = [t * time_scale for t in time_full]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_full_accel, accel_x_full_truncated, 'r-', label='Acceleration X')
plt.plot(time_full_accel, accel_y_full_truncated, 'g-', label='Acceleration Y')
plt.plot(time_full_accel, accel_z_full_truncated, 'b-', label='Acceleration Z')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration vs Time')
plt.legend()
plt.grid(True)
plt.xlim(0, 50)  # Limit x-axis to 50 seconds
plt.savefig('acceleration.png')

plt.figure(figsize=(10, 6))
plt.plot(time_full, vel_x, 'r-', label='Velocity X')
plt.plot(time_full, vel_y, 'g-', label='Velocity Y')
plt.plot(time_full, vel_z, 'b-', label='Velocity Z')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time')
plt.legend()
plt.grid(True)
plt.savefig('velocity.png')

plt.figure(figsize=(10, 6))
plt.plot(time_full, dist_x, 'r-', label='Distance X')
plt.plot(time_full, dist_y, 'g-', label='Distance Y')
plt.plot(time_full, dist_z, 'b-', label='Distance Z')
plt.xlabel('Time (seconds)')
plt.ylabel('Distance (m)')
plt.title(f'Distance vs Time (Apogee: {max(dist_z):.2f} m)')
plt.legend()
plt.grid(True)
plt.savefig('distance.png')

print(f"Total flight time: {time_full[-1]:.2f} seconds")
print(f"Drogue deployment at t = {drogue_deploy_time:.2f} s, altitude = {drogue_deploy_dist:.2f} m")
print(f"Max apogee reached: {max(dist_z):.2f} meters")
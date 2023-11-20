import math
import random
import colorsys
import numpy as np
class Particle:
    def __init__(self, mass, position, velocity, radius, color):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.color = color

    def calculate_distance(self, other_particle):
        x1, y1 = self.position
        x2, y2 = other_particle.position
        dist_x = x2 - x1
        dist_y = y2 - y1
        distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
        angle = math.atan2(dist_y, dist_x)
        return distance, angle

    def barrier(self, wall):
        direction = wall.direction

        if direction == "horizontal":
            
            min_x, max_x = wall.position[0]
            y = wall.position[1]

            if y > 0:
                # Case 1: Some part of particle inside, but center outside
                if self.position[1] > y and self.position[1] < y + self.radius:
                    ext = self.position[1] + self.radius - y
                    self.position[1] = y - ext - self.radius
                    self.velocity[1] = -self.velocity[1]
                    
                # Case 2: Some part of particle outside and center inside
                if self.position[1] < y and self.position[1] + self.radius > y:
                    ext = self.position[1] + self.radius - y
                    self.position[1] = y - ext - self.radius
                    self.velocity[1] = -self.velocity[1]
                    
                # Case 3: Particle outside 
                if self.position[1] > y:
                    ext = self.position[1] - y
                    self.position[1] = y - ext
                    self.velocity[1] = -self.velocity[1]
                    
                
                
            elif y < 0:
                # Case 1: Some part of particle inside, but center outside
                if self.position[1] < y and self.position[1] > y - self.radius:
                    ext = y - self.position[1] + self.radius
                    self.position[1] = y + ext + self.radius
                    self.velocity[1] = -self.velocity[1]
                    
                # Case 2: Some part of particle outside and center inside
                if self.position[1] > y and self.position[1] - self.radius < y:
                    ext = y - self.position[1] + self.radius
                    self.position[1] = y + ext + self.radius
                    self.velocity[1] = -self.velocity[1]
                    
                # Case 3: Particle outside 
                if self.position[1] < y:
                    ext = y - self.position[1]
                    self.position[1] = y + ext
                    self.velocity[1] = -self.velocity[1]
                    
        elif direction == "vertical":
            x = wall.position[0]
            min_y, max_y = wall.position[1]
            if min_y<self.position[1]<max_y:
                if x > 0:
                    # Case 1: Some part of particle inside, but center outside
                    if self.position[0] > x and self.position[0] < x + self.radius:
                        ext = self.position[0] + self.radius - x
                        self.position[0] = x - ext - self.radius
                        self.velocity[0] = -self.velocity[0]
                    # Case 2: Some part of particle outside and center inside
                    if self.position[0] < x and self.position[0] + self.radius > x:
                        ext = self.position[0] + self.radius - x
                        self.position[0] = x - ext - self.radius
                        self.velocity[0] = -self.velocity[0]
                    # Case 3: Particle outside
                    if self.position[0] > x:
                        ext = self.position[0] - x
                        self.position[0] = x - ext
                        self.velocity[0] = -self.velocity[0]

                elif x < 0:
                    # Case 1: Some part of particle inside, but center outside
                    if self.position[0] < x and self.position[0] > x - self.radius:
                        ext = x - self.position[0] + self.radius
                        self.position[0] = x + ext + self.radius
                        self.velocity[0] = -self.velocity[0]
                    # Case 2: Some part of particle outside and center inside
                    if self.position[0] > x and self.position[0] - self.radius < x:
                        ext = x - self.position[0] + self.radius
                        self.position[0] = x + ext + self.radius
                        self.velocity[0] = -self.velocity[0]
                    # Case 3: Particle outside
                    if self.position[0] < x:
                        ext = x - self.position[0]
                        self.position[0] = x + ext
                        self.velocity[0] = -self.velocity[0]
                

        
            

        

        return self.velocity
        
    


    def collision(self, other_particle, min_collision_distance):
        distance, angle = self.calculate_distance(other_particle)
        collision_distance = self.radius + other_particle.radius

        if distance <= collision_distance and distance > min_collision_distance:
            # Step 1: Find unit normal and unit tangent vectors
            normal_vector = [other_particle.position[0] - self.position[0], other_particle.position[1] - self.position[1]]
            normal_magnitude = math.sqrt(normal_vector[0] ** 2 + normal_vector[1] ** 2)
            unit_normal_vector = [normal_vector[0] / normal_magnitude, normal_vector[1] / normal_magnitude]
            unit_tangent_vector = [-unit_normal_vector[1], unit_normal_vector[0]]

            # Step 2: Create initial velocity vectors
            v1 = self.velocity
            v2 = other_particle.velocity

            # Step 3: Project velocities onto normal and tangent vectors
            v1n = unit_normal_vector[0] * v1[0] + unit_normal_vector[1] * v1[1]
            v1t = unit_tangent_vector[0] * v1[0] + unit_tangent_vector[1] * v1[1]
            v2n = unit_normal_vector[0] * v2[0] + unit_normal_vector[1] * v2[1]
            v2t = unit_tangent_vector[0] * v2[0] + unit_tangent_vector[1] * v2[1]

            # Step 4: New tangential velocities (unchanged)
            v1t_after = v1t
            v2t_after = v2t

            # Step 5: New normal velocities (one-dimensional collision)
            m1 = self.mass
            m2 = other_particle.mass
            v1n_after = ((v1n * (m1 - m2)) + (2 * m2 * v2n)) / (m1 + m2)
            v2n_after = ((v2n * (m2 - m1)) + (2 * m1 * v1n)) / (m1 + m2)

            # Step 6: Convert scalar velocities to vectors
            v1n_vector = [unit_normal_vector[0] * v1n_after, unit_normal_vector[1] * v1n_after]
            v1t_vector = [unit_tangent_vector[0] * v1t_after, unit_tangent_vector[1] * v1t_after]
            v2n_vector = [unit_normal_vector[0] * v2n_after, unit_normal_vector[1] * v2n_after]
            v2t_vector = [unit_tangent_vector[0] * v2t_after, unit_tangent_vector[1] * v2t_after]

            # Step 7: Final velocity vectors
            self.velocity = [v1n_vector[0] + v1t_vector[0], v1n_vector[1] + v1t_vector[1]]
            other_particle.velocity = [v2n_vector[0] + v2t_vector[0], v2n_vector[1] + v2t_vector[1]]
            overlap = collision_distance - distance
            displacement = [unit_normal_vector[0] * overlap / 2, unit_normal_vector[1] * overlap / 2]
            self.position[0] -= displacement[0]
            self.position[1] -= displacement[1]
            other_particle.position[0] += displacement[0]
            other_particle.position[1] += displacement[1]


class Wall:
    def __init__(self, position, direction, ):
        self.position = position
        self.direction = direction


def create_box(length, width):
    # Create the four instances of Wall with the specified dimensions
    wall1 = Wall([(width / 2), (-length / 2, length / 2)], 'vertical')
    wall2 = Wall([- (width / 2), (-length / 2, length / 2)], 'vertical')
    wall5 = Wall([ (1), (length / 4, length / 2)], 'vertical')
    wall6 = Wall([ (1), (-length / 2, -length / 4)], 'vertical')

    wall3 = Wall([(-width / 2, width / 2), length / 2], 'horizontal')
    wall4 = Wall([(-width / 2, width / 2), - (length / 2)], 'horizontal')
    
    return wall1, wall2, wall3, wall4





def generate_random_particles(num_particles, max_position, max_speed, mass_range):
    particles = []
    for _ in range(num_particles):
        mass = random.uniform(mass_range[0], mass_range[1])
        position = [random.uniform(-max_position, 0), random.uniform(-max_position, max_position)]
        velocity = [random.uniform(-max_speed, max_speed), random.uniform(-max_speed, max_speed)]
        radius = mass**(1/3)   # Radius is proportional to mass
        color = colorsys.hsv_to_rgb(random.random(), 1, 1)  # Random color for each particle
        particle = Particle(mass, position, velocity, radius, color)
        particles.append(particle)
    return particles

length = 100
width = 100

# Create the box using the create_box function
walls = create_box(length, width)
particles = generate_random_particles(10, 40, 40, (1, 3))

simulation_time = 10
time_step = 0.01
min_collision_distance = 0.001

particle_positions_x = [[] for _ in range(len(particles))]
particle_positions_y = [[] for _ in range(len(particles))]
nb = 0
# Loop 1: Calculate particle positions
for _ in range(int(simulation_time / time_step)):
    
    nb = 0
    for i, particle in enumerate(particles):
        
        particle.position[0] += particle.velocity[0] * time_step
        particle.position[1] += particle.velocity[1] * time_step
        if - width / 2 <= particle.position[0] <= width/2:
            if -length / 2 <= particle.position[1] <= length/2:
                nb += 1



        
        for wall in walls:
            particle.barrier(wall)

        for j, other_particle in enumerate(particles):
            if i != j:
                particle.collision(other_particle, min_collision_distance)

        particle_positions_x[i].append(particle.position[0])
        particle_positions_y[i].append(particle.position[1])
    print(nb)
    
    total_kinetic_energy = 0

    for particle in particles:
        kinetic_energy = 0.5 * particle.mass * (particle.velocity[0]**2 + particle.velocity[1]**2)
        total_kinetic_energy += kinetic_energy

    average_kinetic_energy_per_particle = total_kinetic_energy / len(particles)

# Calculate degrees of freedom for a 2D system
    df = 2 * len(particles) - 2

# Boltzmann constant (J/K)
    kB = 1.380649e-23  # Value in Joules per Kelvin

# Calculate the temperature using the formula
    temperature = average_kinetic_energy_per_particle / (0.5 * kB * df)

# Print or use the calculated temperature as needed
    print("Average Kinetic Energy per Particle:", average_kinetic_energy_per_particle)
    print("Temperature:", temperature)









import matplotlib.pyplot as plt

fig = plt.figure(facecolor='black')
ax = fig.add_subplot(111)
for wall in walls:
    if wall.direction == 'vertical':
        x = wall.position[0]
        y1, y2 = wall.position[1]
        ax.plot([x, x], [y1, y2], 'k-', lw=2)
    elif wall.direction == 'horizontal':
        x1, x2 = wall.position[0]
        y = wall.position[1]
        ax.plot([x1, x2], [y, y], 'k-', lw=2)
plt.xlim(-width/2, width/2)
plt.ylim(-length/2, length/2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Particle Bouncing')
# Generate random colors for each particle
colors = [particle.color for particle in particles]
# Precalculated particle positions
particle_positions = list(zip(particle_positions_x, particle_positions_y))
# Loop: Animate precalculated particle positions
for iteration in range(int(simulation_time / time_step)):
    ax.cla()

    for wall in walls:
        if wall.direction == 'vertical':
            x = wall.position[0]
            y1, y2 = wall.position[1]
            ax.plot([x, x], [y1, y2], 'k-', lw=2)
        elif wall.direction == 'horizontal':
            x1, x2 = wall.position[0]
            y = wall.position[1]
            ax.plot([x1, x2], [y, y], 'k-', lw=2)

    for i, particle in enumerate(particles):
        x = particle_positions_x[i][iteration]
        y = particle_positions_y[i][iteration]
        particle.position = [x, y]

        color = (particle.color[0], particle.color[1], particle.color[2])
        circle = plt.Circle(particle.position, particle.radius, edgecolor='black', facecolor=color)
        ax.add_artist(circle)

    ax.set_xlim(-width/2, width/2)
    ax.set_ylim(-length/2, length/2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Particle Bouncing')
    ax.set_aspect('equal')

    
    plt.text(-0.38, 0.95, f'Number of Particles', transform=ax.transAxes, color='white')
    plt.text(-0.38, 0.9, f'{nb}', transform=ax.transAxes, color='white')
    plt.text(1.05, 0.95, f'Temperature:', transform=ax.transAxes, color='white')
    plt.text(1.05, 0.9, f'{temperature:.2f} K', transform=ax.transAxes, color='white')


    plt.pause(0.01)

plt.show()


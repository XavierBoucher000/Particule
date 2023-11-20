

Particle Class:
Represents a particle with properties such as mass, position, velocity, radius, and color.
Contains methods for calculating distance between particles, handling collisions with walls, and managing particle-particle collisions.
Wall Class:
Represents a wall with properties such as position and direction (horizontal or vertical).
create_box Function:
Generates walls to create a box with specified dimensions.
generate_random_particles Function:
Creates a specified number of particles with random masses, positions, velocities, and colors.
Main Simulation Loop:
Advances the simulation in discrete time steps.
Updates particle positions based on their velocities.
Checks for collisions with walls using the barrier method in the Particle class.
Handles collisions between particles using the collision method in the Particle class.
Stores particle positions over time for later animation.
Visualization:
Uses Matplotlib to visualize the simulation.
Walls are drawn, and particles are represented as circles with different colors.
The simulation is animated to show the movement and collisions of particles.

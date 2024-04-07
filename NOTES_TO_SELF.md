# Container Setup Instructions

This document outlines the necessary steps to set up your environment within the container, including creating a symbolic link and adding a dependency to a configuration file.

## Creating a Symbolic Link

To create a symbolic link inside the container, follow these steps. This link will point to the Isaac Sim installation directory, simplifying access to it from a common workspace directory.

1. Navigate to your workspace directory:

    ```bash
    cd /workspace/orbit
    ```

2. Create a symbolic link named `_isaac_sim` that points to the `/isaac-sim/` directory:

    ```bash
    ln -s /isaac-sim/ _isaac_sim
    ```

By doing this, you can easily refer to the Isaac Sim directory using the `_isaac_sim` shortcut from your workspace.

## Adding a Dependency

To add a new dependency to the Isaac Sim environment, you'll need to modify a configuration file. Specifically, we'll be adding the "omni.isaac.range_sensor" extension.

1. Ensure the configuration file exists by touching the file (this command will create the file if it doesn't exist, but won't modify it if it does):

    ```bash
    touch /isaac-sim/apps/omni.isaac.sim.python.gym.headless.kit
    ```

2. Add the following dependency to the bottom of the file. This step might require using a text editor to open and edit the file:

    ```bash
    "omni.isaac.range_sensor" = {}
    ```

This line registers the `omni.isaac.range_sensor` extension with the Isaac Sim environment, making it available for use in your simulations.

### Note:

- These instructions assume you have the necessary access and permissions within the container to create symbolic links and modify configuration files.
- Be sure to verify the path to the Isaac Sim directory and the location of the configuration file, as these may vary based on your specific installation and version of Isaac Sim.

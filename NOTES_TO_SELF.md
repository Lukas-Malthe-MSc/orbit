# Notes to Self

## Working with Submodules in Git

### Initializing Submodules

After cloning a repository with submodules, you need to initialize and update the submodules. Here's how to do it sequentially:

1. **Initialize Your Submodules**: This sets up your local configuration file.
    ```bash
    git submodule init
    ```

2. **Fetch All Data for the Submodules**:
    ```bash
    git submodule update
    ```

    Or, to initialize and update in one command:
    ```bash
    git submodule update --init
    ```

    For submodules within submodules:
    ```bash
    git submodule update --init --recursive
    ```

## Changing Read/Write Permissions

### For the Orbit Directory:

From the host system, navigate to the directory containing the Orbit project and adjust permissions as necessary. A more secure setting that allows read, write, and execute for the owner, and read and execute for others is:

```bash
chmod 755 -R orbit
chmod 700 ~/.ssh
```
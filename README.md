# Project README

## Overview
This project is designed to assist users in managing files within a directory. It utilizes an agent-based architecture to provide functionalities such as listing files, reading file contents, and ensuring safe operations without accessing restricted files.

## Architecture
The core components of the project include:
- **Agent**: The main entity that processes user tasks and manages goals.
- **Goals**: Defined objectives for the agent, including file management and safety protocols.
- **Environment**: The context in which the agent operates, handling interactions with the file system.
- **Action Registry**: A collection of actions that the agent can perform, such as listing files and reading file contents.

The project is structured into a main module (`main.py`) and a package directory (`game`) that contains the core functionalities.

## Setup
To set up the project, ensure you have Python installed and then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
To run the project, execute the main module:

```bash
python main.py
```

The agent will prompt you for a task, and you can interact with it to manage files in the current directory.
# Titanic ML Project with Metaflow and MLflow

This project demonstrates how to build a machine learning pipeline for the Titanic dataset using Metaflow for workflow orchestration and MLflow for experiment tracking. The project uses a VS Code Dev Container for a consistent development environment.

## Development Environment

This project uses VS Code Dev Containers to provide a consistent Linux-based development environment. This ensures compatibility with Metaflow, which has dependencies that work best on Linux.

### Prerequisites

1. [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. [Visual Studio Code](https://code.visualstudio.com/)
3. [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code

### Getting Started with Dev Container

1. Clone the repository
2. Open the project folder in VS Code
3. When prompted, click "Reopen in Container" or use the command palette (Ctrl+Shift+P) and select "Remote-Containers: Reopen in Container"
4. The container will build with all necessary dependencies (Python, Metaflow, MLflow, etc.)

If you encounter any issues on the first attempt, try the "Retry" option or rebuild the container.

## Quick Start Guide

### 1. Download the Titanic Dataset

Option 1: Using Kaggle API (recommended)
```bash
# Configure your Kaggle API credentials first
mkdir -p ~/.kaggle
echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Create data directory
mkdir -p data

# Download Titanic dataset
kaggle competitions download -c titanic -p ./data
unzip ./data/titanic.zip -d ./data
```

Option 2: Manual download
- Go to [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
- Download the train.csv file
- Place it in a `data` folder in your project directory

### 2. Run the Metaflow Pipeline

```bash
python titanic_flow.py run
```

### 3. View Results in MLflow UI

```bash
mlflow ui
```
Then visit [http://localhost:5000](http://localhost:5000) in your browser. VS Code will automatically forward the port.

## Project Structure

- `titanic_flow.py`: Main Metaflow pipeline file
- `.devcontainer/`: Contains configuration for the development container
- `data/`: Directory containing the Titanic dataset (gitignored)
- `mlruns/`: Directory where MLflow stores experiment data (gitignored)
- `.metaflow/`: Metaflow metadata directory (gitignored)

## Pipeline Steps

1. **Start**: Checks if the data file exists
2. **Load Data**: Loads the Titanic dataset and displays basic information
3. **Preprocess Data**: Performs feature engineering and data cleaning
4. **Prepare Features**: Splits data into training and testing sets
5. **Train Model**: Trains a RandomForest model and tracks with MLflow
6. **End**: Displays summary of the process

## Advanced Features

For more advanced functionality, see the advanced_titanic_flow.py file, which includes:

- Multiple model training in parallel
- Hyperparameter tuning
- Feature selection
- Model comparison
- Simulated deployment

## Troubleshooting

If you encounter issues:

1. **Metaflow Import Errors**: Ensure you're running inside the dev container
2. **Missing Titanic Dataset**: Check that the dataset is in the correct location (`./data/train.csv`)
3. **MLflow UI Not Accessible**: Ensure port forwarding is enabled in VS Code

## Next Steps

Once you've completed this basic pipeline, you can:

1. Try different ML algorithms (e.g., XGBoost, LogisticRegression)
2. Perform hyperparameter tuning
3. Add more features or try feature selection
4. Create a deployment step to serve the model

Happy learning!
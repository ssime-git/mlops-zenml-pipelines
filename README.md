# mlops-zenml-pipelines

# Installation de uv

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Initialisation du projet

```sh
uv init
uv venv
uv sync
```

# Intallation des intégrations

```sh
zenml integration install mlflow -y
zenml integration install sklearn y
```
# Initialisation du projet zenml

```sh
zenml init
```

# Création de la Stack

## Experiment tracker

```sh
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
```

## Création de la stack

```sh
zenml stack register mlflow_stack \
    -e mlflow_tracker \
    -a default \
    -o default

# pour lister les stack
zenml stack list

# activer la stack
zenml stack set mlflow_stack
```
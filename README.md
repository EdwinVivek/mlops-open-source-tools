In this project, we will develop a machine learning workflow utilizing the MLOps pipeline. We will employ some of the open-source tools to construct the MLOps pipeline. This pipeline will encompass the full lifecycle of machine learning model development, which includes data preprocessing, model training, feature engineering, model monitoring, deployment, and implementing CI/CD pipelines.

When discussing the automation of the MLOps pipeline, it is important to highlight the role of Continuous Monitoring. Therefore, we will incorporate a feedback loop to complete the MLOps cycle through Continuous monitoring and re-training of the ML model.

In this project, we will utilize the following open source tools to establish the MLOps pipeline:

Feast
Mlflow
BentoML
Docker
Evidently
Apache Airflow
Project Overview:

The project involves:

1. **Data Preprocessing & Ingestion:** Preparing and processing data with Pandas
2. **Feature Store & Feature Engineering:** Moving the process data to feature store to store and organize them
3. **Experiment tracking & Model Registry:** Manage tracking, versioning, and training of a Scikit-learn model.
4. **Model Evaluation & serving:** Deploy the model through an API.
5. **Model Monitoring:** Assess data drift, concept drift, and model performance using reports and dashboards.
6. **Containerization:** Use Docker to containerize the model
7. **Continuous Integration:** Implement CI to initiate model training with every modification to the source code.
8. **Model monitoring & retraining:** Consistently evaluate the model with new data and retrain as necessary.

For detailed explanation on each steps refer, 

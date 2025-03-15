from pycaret.anomaly import AnomalyExperiment
from pycaret.datasets import get_data

# Load dataset
data = get_data('anomaly')

# Initialize experiment
exp = AnomalyExperiment()
exp.setup(data, session_id=123, normalize=True, transformation=True)  # Added normalization and transformation

# Train multiple models
iforest = exp.create_model('iforest')
knn = exp.create_model('knn')
lof = exp.create_model('lof')
cblof = exp.create_model('cblof')  # Added Cluster-Based Local Outlier Factor
svm = exp.create_model('svm')  # Added One-Class SVM

# Evaluate models
best_model = exp.compare_models()  # Compare different anomaly detection models and select the best one

# Assign anomaly labels
iforest_results = exp.assign_model(iforest)
knn_results = exp.assign_model(knn)
lof_results = exp.assign_model(lof)
cblof_results = exp.assign_model(cblof)
svm_results = exp.assign_model(svm)

# # Plot results for visualization
# exp.plot_model(iforest, plot='tsne')
# exp.plot_model(knn, plot='umap')
# exp.plot_model(lof, plot='distribution')
# exp.plot_model(cblof, plot='tsne')
# exp.plot_model(svm, plot='distribution')

# Save models and processed datasets
# exp.save_model(iforest, 'iforest_pipeline')
# exp.save_model(knn, 'knn_pipeline')
# exp.save_model(lof, 'lof_pipeline')
# exp.save_model(cblof, 'cblof_pipeline')
# exp.save_model(svm, 'svm_pipeline')

# Save processed datasets
# Use assign to record the anomalies
iforest_results.to_csv('iforest_results.csv', index=False)
knn_results.to_csv('knn_results.csv', index=False)
lof_results.to_csv('lof_results.csv', index=False)
cblof_results.to_csv('cblof_results.csv', index=False)
svm_results.to_csv('svm_results.csv', index=False)

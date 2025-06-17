# Raikura Example Notebook

# In[1]: Import core libraries
import pandas as pd
from raikura import AutoMLPipeline, DataLoader

# In[2]: Load a sample dataset
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df['target'] = data.target
X = df.drop('target', axis=1)
y = df['target']

# In[3]: Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In[4]: Create and configure AutoML pipeline
pipe = AutoMLPipeline(model_type="random_forest")
pipe.configure(
    preprocessing={
        "scaling": True,
        "expand_polynomial": True,
        "poly_degree": 2
    },
    evaluation={
        "metrics": ["accuracy", "f1", "roc_auc"]
    }
)

# In[5]: Train the model
pipe.train(X_train, y_train)

# In[6]: Evaluate the model on test set
results = pipe.evaluate(X_test, y_test)
print("Evaluation Results:")
print(results)

# In[7]: Explain the model
from raikura import Explainability
explainer = Explainability(model=pipe.model, data=X_train)
shap_values = explainer.shap_summary_plot(X_train)

# In[8]: Save model and config
pipe.save_model("outputs/model.pkl")
pipe.save_config("outputs/config.yaml")

# In[9]: Load and predict
pipe.load_model("outputs/model.pkl")
preds = pipe.predict(X_test)
print("Predictions:", preds[:10])

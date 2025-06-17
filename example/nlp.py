# Raikura NLP Transformer Classification Notebook

# In[1]: Import required libraries
import pandas as pd
from raikura import AutoMLPipeline, DataLoader

# In[2]: Load a sample text dataset
data = {
    "text": [
        "I love this product! Absolutely amazing.",
        "Terrible service, very disappointed.",
        "Had a great experience with the app.",
        "This is the worst thing I’ve ever bought.",
        "I’m very happy with the results."
    ],
    "label": [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
X = df[["text"]]
y = df["label"]

# In[3]: Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In[4]: Configure transformer classifier pipeline
pipeline = AutoMLPipeline(model_type="bert_classifier")
pipeline.configure(
    preprocessing={
        "text_column": "text",
        "max_length": 128
    },
    model_params={
        "pretrained_model": "bert-base-uncased",
        "epochs": 3,
        "batch_size": 8
    },
    evaluation={"metrics": ["accuracy", "f1"]}
)

# In[5]: Train model
pipeline.train(X_train, y_train)

# In[6]: Evaluate on test set
results = pipeline.evaluate(X_test, y_test)
print("Evaluation Results:")
print(results)

# In[7]: Predict on new samples
sample = pd.DataFrame({"text": ["This changed my life!"]})
preds = pipeline.predict(sample)
print("Prediction:", preds)

# In[8]: Save model
pipeline.save_model("outputs/transformer_model.pkl")

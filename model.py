import joblib

def train_model():

    # Save model
    joblib.dump(model, 'models/trained_model.pkl')
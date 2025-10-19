def load_model(filename):
    import joblib
    model = joblib.load(filename)
    return model

def generate_c_code(model):
    from sklearn.linear_model import LogisticRegression
 
    is_logistic = isinstance(model, LogisticRegression)

    coefficients = model.coef_
    intercept = model.intercept_
    
    if len(coefficients.shape) > 1:
        coefficients = coefficients[0]
    if hasattr(intercept, '__len__'):
        intercept = intercept[0]
    
    n_features = len(coefficients)
    
    code = "#include <stdio.h>\n"
    code += "#include <math.h>\n\n"
    
    if is_logistic:
        code += "float sigmoid(float x) {\n"
        code += "    return 1.0 / (1.0 + exp(-x));\n"
        code += "}\n\n"
    
    code += "float prediction(float *features, int n_feature) {\n"
    code += "    float result = {:.6f};\n".format(intercept)
    for i in range(n_features):
        code += "    result += {:.6f} * features[{}];\n".format(coefficients[i], i)
    
    if is_logistic:
        code += "    return sigmoid(result);\n"
    else:
        code += "    return result;\n"
    code += "}\n\n"
    
    code += "int main() {\n"
    code += "    float features[{}] = {{1.0, 2.0, 0.0}}; // Example features\n".format(n_features)
    code += "    float pred = prediction(features, {});\n".format(n_features)
    
    if is_logistic:
        code += "    printf(\"Prediction (probability): %f\\n\", pred);\n"
        code += "    printf(\"Class: %d\\n\", pred >= 0.5 ? 1 : 0);\n"
    else:
        code += "    printf(\"Prediction: %f\\n\", pred);\n"
    
    code += "    return 0;\n"
    code += "}\n"
    
    return code

def save_c_code(code, filename):
    with open(filename, 'w') as f:
        f.write(code)

def main():
    # SI vous voulez tester mon code Vous pouvez changer entre 'regression.joblib' et 'logistic_regression.joblib'
    model_file = 'regression.joblib'  # ou 'logistic_regression.joblib' pour la régression logistique
    model = load_model(model_file)
    c_code = generate_c_code(model)
    c_filename = 'model.c'
    save_c_code(c_code, c_filename)
    print(f"C code saved to {c_filename}")
    print("To compile the code, run:")
    print(f"gcc -o model {c_filename} -lm")  # -lm pour la bibliothèque math (exp)

def build_model():
    import pandas as pd 
    from sklearn.linear_model import LinearRegression
    import joblib
    df = pd.read_csv('houses.csv')
    X = df[['size', 'nb_rooms', 'garden']]
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "regression.joblib")

def build_logistic_model():
    """Crée un modèle de régression logistique pour la classification"""
    import pandas as pd 
    from sklearn.linear_model import LogisticRegression
    import joblib
    
    df = pd.read_csv('houses.csv')
    X = df[['size', 'nb_rooms', 'garden']]
    # Créer une variable binaire (par exemple, prix > médiane)
    y = (df['price'] > df['price'].median()).astype(int)
    
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "logistic_regression.joblib")
    print("Logistic regression model saved to logistic_regression.joblib")

if __name__ == "__main__":
    main()
def load_model(filename):
    import joblib
    model = joblib.load(filename)
    return model

def generate_tree_c_code(tree, feature_names=None, indent=0):
    """Génère récursivement le code C pour un arbre de décision"""
    from sklearn.tree import _tree
    
    tree_ = tree.tree_
    feature_name = tree_.feature
    threshold = tree_.threshold
    
    def recurse(node, depth):
        indent_str = "    " * depth
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_idx = tree_.feature[node]
            threshold_val = tree_.threshold[node]
            
            code = f"{indent_str}if (features[{feature_idx}] <= {threshold_val:.6f}) {{\n"
            code += recurse(tree_.children_left[node], depth + 1)
            code += f"{indent_str}}} else {{\n"
            code += recurse(tree_.children_right[node], depth + 1)
            code += f"{indent_str}}}\n"
            return code
        else:
            value = tree_.value[node]
            if len(value[0]) == 1:
                return f"{indent_str}return {value[0][0]:.6f};\n"
            else:
                class_idx = value[0].argmax()
                return f"{indent_str}return {class_idx};\n"
    
    return recurse(0, indent)

def generate_c_code(model):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    
    is_logistic = isinstance(model, LogisticRegression)
    is_tree = isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor))
    
    code = "#include <stdio.h>\n"
    code += "#include <math.h>\n\n"
    
    if is_tree:
        code += "float prediction(float *features, int n_features) {\n"
        code += generate_tree_c_code(model, indent=1)
        code += "}\n\n"
        
        n_features = model.n_features_in_
        
        code += "int main() {\n"
        code += "    float features[{}] = {{1.0, 2.0, 0.0}}; // Example features\n".format(n_features)
        code += "    float pred = prediction(features, {});\n".format(n_features)
        
        if isinstance(model, DecisionTreeClassifier):
            code += "    printf(\"Predicted class: %d\\n\", (int)pred);\n"
        else:
            code += "    printf(\"Prediction: %f\\n\", pred);\n"
        
        code += "    return 0;\n"
        code += "}\n"
    else:
        
        coefficients = model.coef_
        intercept = model.intercept_
        
        if len(coefficients.shape) > 1:
            coefficients = coefficients[0]
        if hasattr(intercept, '__len__'):
            intercept = intercept[0]
        
        n_features = len(coefficients)
        
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
    # Vous pouvez changer entre:
    # 'regression.joblib' - régression linéaire
    # 'logistic_regression.joblib' - régression logistique
    # 'decision_tree_classifier.joblib' - arbre de décision (classification)
    # 'decision_tree_regressor.joblib' - arbre de décision (régression)
    model_file = 'decision_tree_classifier.joblib'
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

def build_decision_tree_classifier():
    """Crée un modèle d'arbre de décision pour la classification"""
    import pandas as pd 
    from sklearn.tree import DecisionTreeClassifier
    import joblib
    
    df = pd.read_csv('houses.csv')
    X = df[['size', 'nb_rooms', 'garden']]
    # Créer une variable binaire (par exemple, prix > médiane)
    y = (df['price'] > df['price'].median()).astype(int)
    
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "decision_tree_classifier.joblib")
    print("Decision tree classifier model saved to decision_tree_classifier.joblib")

def build_decision_tree_regressor():
    """Crée un modèle d'arbre de décision pour la régression"""
    import pandas as pd 
    from sklearn.tree import DecisionTreeRegressor
    import joblib
    
    df = pd.read_csv('houses.csv')
    X = df[['size', 'nb_rooms', 'garden']]
    y = df['price']
    
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "decision_tree_regressor.joblib")
    print("Decision tree regressor model saved to decision_tree_regressor.joblib")

if __name__ == "__main__":
    main()
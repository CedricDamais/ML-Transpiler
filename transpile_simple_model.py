"""
oder un script python transpile_simple_model.py qui :
- charge un fichier joblib contenant une régression linéaire entraînée
- récupère les valeurs de coefficients 
- génère une chaîne de caractère contenant le code C permettant de calculer la prédiction du modèle (float prediction(float *features, int n_feature) )avec les valeur du coefficient
- génère une fonction main qui permet d'appeler prediction sur une donnée définié par un tableau statique de votre choix. 
- sauvegarde le code c généré dans un fichier.c 
- et affiche la commande de compilation à lancer pour le compiler ou le compile directement.
 
2. Entraîner une régression linéaire simple sur un dataset simple (par exemple houses.csv) et le sauvegarder

3. Lancer le script transpile_simple_model et compiler le fichier C généré. Vérifier que les prédictions produites par votre code  sont conformes au model.predict
"""

def load_model(filename):
    import joblib
    model = joblib.load(filename)
    return model

def generate_c_code(model):
    coefficients = model.coef_
    intercept = model.intercept_
    n_features = len(coefficients)
    
    code = "#include <stdio.h>\n\n"
    code += "float prediction(float *features, int n_feature) {\n"
    code += "    float result = {:.6f};\n".format(intercept)
    for i in range(n_features):
        code += "    result += {:.6f} * features[{}];\n".format(coefficients[i], i)
    code += "    return result;\n"
    code += "}\n\n"
    
    code += "int main() {\n"
    code += "    float features[{}] = {{1.0, 2.0, 0.0}}; // Example features\n".format(n_features)
    code += "    float pred = prediction(features, {});\n".format(n_features)
    code += "    printf(\"Prediction: %f\\n\", pred);\n"
    code += "    return 0;\n"
    code += "}\n"
    
    return code

def save_c_code(code, filename):
    with open(filename, 'w') as f:
        f.write(code)

def main():
    model = load_model('regression.joblib')
    c_code = generate_c_code(model)
    c_filename = 'model.c'
    save_c_code(c_code, c_filename)
    print(f"C code saved to {c_filename}")
    print("To compile the code, run:")
    print(f"gcc -o model {c_filename}")

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

if __name__ == "__main__":
    main()
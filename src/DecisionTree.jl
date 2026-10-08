using CSV
using DataFrames
using DecisionTree

data = CSV.read("data/bc_data.csv", DataFrame)

# Vérification de la database 

#first(data, 5)
describe(data) # Vérification des données manquantes 
size(data)
names(data)

X = Matrix(data[:, 1:9]) # matrice des mesures médicales 
y = string.(data[:, 10]) # vecteur contenant le diagnostic réel de chaque patiente ( Classification )
# string pour forcer la classification ( résultat fini = malade ou pas malade) et éviter une régression 

using Random
using Statistics #  pour la fonction mean()

# 1. Reproductibilité
Random.seed!(1) #fonction seed : point de départ du tirage aléatoire  

# 2. Découpage Train(entrainement) / Test (70% - 30%)
n = length(y) #nombre total de patientes
indices = randperm(n) # Mélange aléatoire des indices [1,.., 116] => [84, 12,45,...]

n_train = round(Int, 0.7 * n) # calcul et arrondi les 70% de l'effectif 
train_idx = indices[1:n_train] # constitution du groupe d'entrainement 
test_idx = indices[n_train+1:end] # constitution du groupe test 

X_train, y_train = X[train_idx, :], y[train_idx]
X_test, y_test   = X[test_idx, :], y[test_idx]

# 3. Entraînement de l'arbre
# AIDE IA
# build_tree(labels, features, n_subfeatures, max_depth)
# n_subfeatures = 0 signifie qu'on teste toutes les 9 variables à chaque coupure

# On teste avec différentes profondeurs de l'arbre pour voir lequel est le plus optimisé
model = build_tree(y_train, X_train, 0, 3)  
# model = build_tree(y_train, X_train, 0, 2) 
# model = build_tree(y_train, X_train, 0, 4)   

# 4. Affichage de l'arbre
println("--- RÈGLES DE DÉCISION DE L'ARBRE ---")
print_tree(model)


###### PARTIE TEST ######

# 5. Prédiction sur le jeu de test
y_pred = apply_tree(model, X_test)

# 6. Évaluation globale
acc = mean(y_pred .== y_test)
println("Taux de bonnes prédictions (Accuracy) : ", round(acc * 100, digits=2), " %")

# 7. Matrice de confusion
# Permet de repérer directement les faux négatifs
cm = confusion_matrix(y_test, y_pred)
println("\nMatrice de confusion :")
println(cm)


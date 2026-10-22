using Pkg; 
Pkg.add("GLM"); 
using CSV
using  DataFrames
using GLM

# 1. Chargement des données
df_reg   = CSV.read(normpath(joinpath(@__DIR__, "..", "data", "coimbra_reg.csv")), DataFrame)
folds_df = CSV.read(normpath(joinpath(@__DIR__, "..", "data", "folds.csv")), DataFrame)

# Vérification de la database 
describe(df_reg)

# Récupération de la colonne des plis
folds = folds_df.fold

#Liste du taux d'erreurs par pli 
taux_erreur_pli = [] 



                            # DECOUPAGE EN PLI
for k in 1:5 
    lignes_train = findall(folds .!= k) # Lignes des patientes du groupe d'entraînement (tous les autres plis sauf k)
    lignes_test  = findall(folds .== k) # Lignes des patientes du groupe test (celles du pli k)

# Séparation train/test 
    train_data = df_reg[lignes_train, :]
    test_data  = df_reg[lignes_test, :]


# Entraînement du modèle de régression linéaire
# On cherche à prédire logLeptin en fonction de l'age, de l'IMC, glucose...
modele = lm(@formula(logLeptin ~ Age + BMI + Glucose + Insulin + Adiponectin + Resistin + MCP1), train_data)


# Prédiction sur les patientes du pli test
predictions = predict(modele, test_data)


# Calcul de l'erreur quadratique moyenne (RMSE) pour chaque pli
vraies_valeurs = test_data.logLeptin
taux_erreur_pli_k = sqrt(mean((vraies_valeurs .- predictions).^2)) 

push!(taux_erreur_pli, taux_erreur_pli_k)
println("Le taux d'erreur du pli $k est  : ", round(taux_erreur_pli_k, digits=4))

end

println("Taux d'erreur moyen :" , round(mean(taux_erreur_pli), digits=4), " (± ", round(std(taux_erreur_pli, digits=4), ")"))


println("Estimation sur l'ensemble du jeu de données" : )
modele_global = lm(@formula(logLeptin ~ Age + BMI + Glucose + Insulin + Adiponectin + Resistin + MCP1), df_reg)
println(modele_global)


println("R² du modèle : ", round(r2(modele_global), digits=4))







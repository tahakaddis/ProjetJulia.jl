using CSV
using DataFrames
using DecisionTree

data = CSV.read("data/bc_data.csv", DataFrame)

# Vérification de la database 

first(data, 5)
size(data)
names(data)

X = Matrix(data[:, 1:9])
y = Vector(data[:, 10])


#Délimitation Partie Entrainement / Partie Test 
train, test = partition(1:length(y), 0.7, shuffle=true)


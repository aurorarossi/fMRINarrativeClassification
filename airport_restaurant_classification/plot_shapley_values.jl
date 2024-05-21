using JLD2
include("plots/plot.jl")

data = load("airport_restaurant_classification/data/shapleyvalues7_15retraining.jld2")
p = shapvalues_plot(data, L"Content classification$$","png")
save("airport_restaurant_classification/plots/shapleyvalues7_15retrainingAR.png",p)

data17 = load("airport_restaurant_classification/data/shapleyvalues17_15retraining.jld2")
p17 = shapvalues17_plot(data17, L"Content classification$$","png")
save("airport_restaurant_classification/plots/shapleyvalues17_15retrainingAR.png",p17)
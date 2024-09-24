using JLD2
include("../plots/plot.jl")

data = load("airport_restaurant_classification/data/shapleyvalues7_15retraining.jld2")
p = shapvalues_plot(data, L"Content classification$$", "png")
save("airport_restaurant_classification/plots/shapleyvalues7_15retrainingAR.png", p)

data17 = load("airport_restaurant_classification/data/shapleyvalues17_15retraining.jld2")
p17 = shapvalues17_plot(data17, L"Content classification$$", "png")
save("airport_restaurant_classification/plots/shapleyvalues17_15retrainingAR.png", p17)

data100_5 = load("airport_restaurant_classification/data/shapleyvalues100_5retraining.jld2")
data100_10 = load("airport_restaurant_classification/data/shapleyvalues100_10retraining.jld2")
d5 = data100_5["d"]
d10 = data100_10["d"]

data100 = Dict(key => vcat(d5[key], d10[key]) for key in keys(d5))

p100 = shapvalues100_plot(data100, L"Content classification$$", "png")
save("airport_restaurant_classification/plots/shapleyvalues100_15retrainingAR.png", p100)

datadesikan = load("airport_restaurant_classification/data/shapleyvalues70_desikan_15retraining.jld2")
pdesikan = shapvaluesdesikan_plot(datadesikan, L"Content classification$$", "png")
save("airport_restaurant_classification/plots/shapleyvalues70_desikan_15retrainingAR.png", pdesikan)

datadestrieux = load("airport_restaurant_classification/data/shapleyvalues75_destrieux_15retraining.jld2")
pdestrieux = shapvaluesdestrieux_plot(datadestrieux, L"Content classification$$", "png")
save("airport_restaurant_classification/plots/shapleyvalues75_destrieux_15retrainingAR.png", pdestrieux)
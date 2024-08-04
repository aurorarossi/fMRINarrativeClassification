using JLD2
include("../plots/plot.jl")
include("../t-test/t-test_17.jl")
include("../t-test/t-test.jl")

data7 = load("airport_restaurant_classification/data/shapleyvalues7_15retraining.jld2")
p = shapvalues_plot(data, L"Content classification$$","png")
save("airport_restaurant_classification/plots/new/shapleyvalues7_15retrainingAR.png",p)

data17 = load("airport_restaurant_classification/data/shapleyvalues17_15retraining.jld2")
p17 = shapvalues17_plot(data17, L"Content classification$$","png")
save("airport_restaurant_classification/plots/new/shapleyvalues17_15retraining.png",p17)

data17 = load("airport_restaurant_classification/data/shapleyvalues17_15retraining_500sample.jld2")
p17 = shapvalues17_plot(data17, L"Content classification 500 samples$$","png")
save("airport_restaurant_classification/plots/new/shapleyvalues17_15retrainingAR_500sample.png",p17)


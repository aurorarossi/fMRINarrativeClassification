using JLD2
include("../plots/plot.jl")

# data7 = load("airport_restaurant_classification/data/shapleyvalues7_15retraining.jld2")
# p = shapvalues_plot(data, L"Content classification$$","png")
# save("airport_restaurant_classification/plots/shapleyvalues7_15retrainingAR.png",p)

# data17 = load("airport_restaurant_classification/data/shapleyvalues17_15retraining.jld2")
# p17 = shapvalues17_plot(data17, L"Content classification$$","png")
# save("airport_restaurant_classification/plots/shapleyvalues17_15retraining.png",p17)

# data17 = load("airport_restaurant_classification/data/shapleyvalues17_15retraining_500sample.jld2")
# p17 = shapvalues17_plot(data17, L"Content classification 500 samples$$","png")
# save("airport_restaurant_classification/plots/shapleyvalues17_15retrainingAR_500sample.png",p17)

# data = load("airport_restaurant_classification/data/shapleyvalues100_5retraining.jld2")
# p = shapvalues100_plot(data, L"Content classification$$","png")
# display(p)
# save("airport_restaurant_classification/plots/shapleyvalues100_5retrainingMA.png",p)

for i in 1:5
    data = JLD2.load("movie_audio_classification/data/shapleyvalues10random_15retraining_$(i).jld2")
    p = shapvaluesrandom_plot(data, L"Content classification$$","png")
end
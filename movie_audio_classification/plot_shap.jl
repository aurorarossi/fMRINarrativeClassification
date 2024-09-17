using Random, CairoMakie, Statistics, LaTeXStrings
using JLD2 
include("../plots/plot.jl")

# data7 = load("movie_audio_classification/data/shapleyvalues7_15retraining_changeshuffling.jld2")
# p = shapvalues_plot(data7, L"Modality classification$$","png")
# display(p)
# save("movie_audio_classification/plots/shapleyvalues7_15retrainingMA.png",p)

# data = load("movie_audio_classification/data/shapleyvalues17_15retraining.jld2")
# p = shapvalues17_plot(data, L"Modality classification$$","png")
# display(p)
# save("movie_audio_classification/plots/shapleyvalues17_15retrainingMA.png",p)

data = load("movie_audio_classification/data/shapleyvalues100_5retraining.jld2")
p = shapvalues100_plot(data, L"Modality classification$$","png")
display(p)
save("movie_audio_classification/plots/shapleyvalues100_5retrainingMA.png",p)


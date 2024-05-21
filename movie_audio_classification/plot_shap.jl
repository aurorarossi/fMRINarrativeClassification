using Random, CairoMakie, Statistics, LaTeXStrings
using JLD2 
include("plots/plot.jl")

data = load("movie_audio_classification/data/shapleyvalues7_15retraining.jld2")
p = shapvalues_plot(data, L"Modality classification$$","png")
save("movie_audio_classification/plots/shapleyvalues7_15retrainingMA.png",p)

data = load("movie_audio_classification/data/shapleyvalues17_15retraining.jld2")
p = shapvalues17_plot(data, L"Modality classification$$","png")
save("movie_audio_classification/plots/shapleyvalues17_15retrainingMA.png",p)
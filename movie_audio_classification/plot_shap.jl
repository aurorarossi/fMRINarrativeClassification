using Random, CairoMakie, Statistics, LaTeXStrings
using JLD2 
include("/user/aurossi/home/fMRINarrativeClassification/plots/plot.jl")
include("../plots/plot.jl")
include("../t-test/t-test_17.jl")
include("../t-test/t-test.jl")

data = load("movie_audio_classification/data/shapleyvalues7_15retraining_changeshuffling.jld2")
p = shapvalues_plot(data, L"Modality classification$$","png", t_test7(data))
display(p)
save("movie_audio_classification/plots/new/shapleyvalues7_15retrainingMA.png",p)

data = load("movie_audio_classification/data/shapleyvalues17_15retraining.jld2")
p = shapvalues17_plot(data, L"Modality classification$$","png", t_test17(data))
display(p)
save("movie_audio_classification/plots/new/shapleyvalues17_15retrainingMA.png",p)
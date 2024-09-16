using Random, CairoMakie, Statistics, LaTeXStrings
using JLD2 

include("../plots/plot.jl")


# data7 = load("movie_audio_classification/data/shapleyvalues7_15retraining_changeshuffling.jld2")
# p = shapvalues_plot(data7, L"Modality classification$$","png")
# display(p)
# save("movie_audio_classification/plots/new/shapleyvalues7_15retrainingMA.png",p)

# data = load("movie_audio_classification/data/shapleyvalues17_15retraining.jld2")
# p = shapvalues17_plot(data, L"Modality classification$$","png")
# display(p)
# save("movie_audio_classification/plots/new/shapleyvalues17_15retrainingMA.png",p)
maxs = zeros(5)
mins = zeros(5)
mean_val = zeros(5*7)
k =1
for i in 1:5
    data = JLD2.load("movie_audio_classification/data/shapleyvalues10random_15retraining_$(i).jld2")
   # p = shapvaluesrandom_plot(data, L"Content classification$$","png")
   data= data["d"]
    for key in  ["r1", "r2" , "r3", "r4", "r5", "r6", "r7"]
        global k
        mean_val[k] = mean(data[key])
        k +=1
    end
end


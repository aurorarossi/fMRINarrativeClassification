using JLD2 
include("../plots/plot.jl")


data = load("movie_audio_classification/data/shapleyvalues100_5retraining.jld2")
p = shapvalues100_plot(data, L"Modality classification$$","png")
display(p)
save("movie_audio_classification/plots/shapleyvalues100_5retrainingMA.png",p)
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

data = load("movie_audio_classification/data/shapleyvalues7_15retraining.jld2")
p = shapvalues_plot(data, L"Modality classification$$","png")
save("movie_audio_classification/plots/shapleyvalues7_15retrainingMA.png", p)

data17 = load("movie_audio_classification/data/shapleyvalues17_15retraining.jld2")
p17 = shapvalues17_plot(data17, L"Modality classification$$","png")
save("movie_audio_classification/plots/shapleyvalues17_15retrainingMA.png", p17)

data100_5 = load("movie_audio_classification/data/shapleyvalues100_5retraining.jld2")
data100_10 = load("movie_audio_classification/data/shapleyvalues100_10retraining.jld2")
d5 = data100_5["d"]
d10 = data100_10["d"]

data100 = Dict(key => vcat(d5[key], d10[key]) for key in keys(d5))

p100 = shapvalues100_plot(data100, L"Modality classification$$","png")
save("movie_audio_classification/plots/shapleyvalues100_15retrainingMA.png", p100)

datadesikan = load("movie_audio_classification/data/shapleyvalues70_desikan_15retraining.jld2")
pdesikan = shapvaluesdesikan_plot(datadesikan, L"Modality classification$$","png")
save("movie_audio_classification/plots/shapleyvalues70_desikan_15retrainingMA.png", pdesikan)

datadestrieux = load("movie_audio_classification/data/shapleyvalues75_destrieux_15retraining.jld2")
pdestrieux = shapvaluesdestrieux_plot(datadestrieux, L"Modality classification$$","png")
save("movie_audio_classification/plots/shapleyvalues75_destrieux_15retrainingMA.png", pdestrieux)
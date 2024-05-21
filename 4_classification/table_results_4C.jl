using JLD2, Statistics
include("utils.jl")

data4C = load("4_classification/data/train_7.jld2")
data4Cpermuted = load("4_classification/data/train_7_permutedtimes.jld2")

data4C = data4C["d"]
data4Cpermuted = data4Cpermuted["d"]
precision4C1 = []
recall4C1 = []
precision4C2 = []
recall4C2 = []
precision4C3 = []
recall4C3 = []
precision4C4 = []
recall4C4 = []

for i in 1:15
    push!(precision4C1, precision(data4C["tp"][i][1], data4C["fp"][i][1]))
    push!(recall4C1, recall(data4C["tp"][i][1], data4C["fn"][i][1]))
    push!(precision4C2, precision(data4C["tp"][i][2], data4C["fp"][i][2]))
    push!(recall4C2, recall(data4C["tp"][i][2], data4C["fn"][i][2]))
    push!(precision4C3, precision(data4C["tp"][i][3], data4C["fp"][i][3]))
    push!(recall4C3, recall(data4C["tp"][i][3], data4C["fn"][i][3]))
    push!(precision4C4, precision(data4C["tp"][i][4], data4C["fp"][i][4]))
    push!(recall4C4, recall(data4C["tp"][i][4], data4C["fn"][i][4]))
end


precision4C = [mean(precision4C1), mean(precision4C2), mean(precision4C3), mean(precision4C4)]

recall4C = [mean(recall4C1), mean(recall4C2), mean(recall4C3), mean(recall4C4)]

stdprecision4C = [std(precision4C1), std(precision4C2), std(precision4C3), std(precision4C4)]

stdrecall4C = [std(recall4C1), std(recall4C2), std(recall4C3), std(recall4C4)]

mp = mean(precision4C) * 100
stp = mean(stdprecision4C) * 100

mre = mean(recall4C) * 100
stre = std(recall4C) * 100

f1score4C = f1_score.(precision4C, recall4C)
stdf1score4C = f1_score.(stdprecision4C, stdrecall4C)

println(mean(data4C["MODEL"]))
println(std(data4C["MODEL"]))
println(mp)
println(stp)
println(mre)
println(stre)
println(mean(f1score4C) * 100)
println(mean(stdf1score4C) * 100)
println(mean(data4Cpermuted["MODEL"]))
println(std(data4Cpermuted["MODEL"]))


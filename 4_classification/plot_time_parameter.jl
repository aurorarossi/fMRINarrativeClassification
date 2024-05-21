using JLD2, CairoMakie, Statistics, LaTeXStrings

data1 = JLD2.load("4_classification/data/train_7_1_8.jld2", "d")["MODEL"]
data2 = JLD2.load("4_classification/data/train_7_2_7.jld2", "d")["MODEL"]
data3 = JLD2.load("4_classification/data/train_7_3_6.jld2", "d")["MODEL"]
data4 = JLD2.load("4_classification/data/train_7_4_5.jld2", "d")["MODEL"]
data5 = JLD2.load("4_classification/data/train_7_5_4.jld2", "d")["MODEL"]
data6 = JLD2.load("4_classification/data/train_7_6_3.jld2", "d")["MODEL"]
data7 = JLD2.load("4_classification/data/train_7_7_2.jld2", "d")["MODEL"]
data8 = JLD2.load("4_classification/data/train_7_8_1.jld2", "d")["MODEL"]

mean1 = mean(data1)
mean2 = mean(data2)
mean3 = mean(data3)
mean4 = mean(data4)
mean5 = mean(data5)
mean6 = mean(data6)
mean7 = mean(data7)
mean8 = mean(data8)

std1 = std(data1)
std2 = std(data2)
std3 = std(data3)
std4 = std(data4)
std5 = std(data5)
std6 = std(data6)
std7 = std(data7)
std8 = std(data8)

means = [mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8]

stds = [std1, std2, std3, std4, std5, std6, std7, std8]

fontsize_theme = Theme(fontsize=20)
set_theme!(fontsize_theme)

color = categorical_colors(:RdBu_4, 4)

fig = Figure(resolution=(800, 600))

ax = Axis(fig[1, 1], 
    xlabel=L"\tau", 
    ylabel=L"Accuracy$$", 
    title=L"Modality and content classification$$",
    xticks=(1:8, [L"1", L"2", L"3", L"4", L"5", L"6", L"7", L"8"]),
    yticks=(25:25:100, [L"25", L"50", L"75", L"100"]))

ylims!(25, 100)

scatter!(ax, 1:8, means, markersize=20, color=color[end], label="Accuracy")
errorbars!(collect(1:8), means, stds, whiskerwidth=10, color=:black)

save("4_classification/plots/plot_time_parameter.png", fig, px_per_unit=2)
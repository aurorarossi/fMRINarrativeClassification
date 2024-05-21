using JLD2
include("plots/plot.jl")

data = load("/4_classification/data/shapleyvalues7_15retraining.jld2")
p = shapvalues_plot(data, L"Combined Modality and Content classiﬁcation$$","png")
save("/4_classification/plots/shapleyvalues7_15retraining4C.png",p)

data17 = load("/4_classification/data/shapleyvalues17_15retraining.jld2")
p17 = shapvalues17_plot(data17, L"Combined Modality and Content classiﬁcation$$","png")
save("/4_classification/plots/shapleyvalues17_15retraining4C.png",p17)
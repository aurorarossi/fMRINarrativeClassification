using JLD2, Statistics

file = load("/user/aurossi/home/fMRINarrativeClassification/4_classification/data/intra_inter_15retraining.jld2")
d = file["d"]
mean(d["acc_intra"])
d["std_acc_intra"] 
d["std_acc_inter"] 
d["totalstd"] 


println("intra")
println(mean(d["std_acc_intra"]))

println("inter")
println(mean(d["std_acc_inter"]))

println("totalstd")
println(mean(d["totalstd"]))

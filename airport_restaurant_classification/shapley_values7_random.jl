using CUDA, Statistics, JLD2
include("../utils.jl")
include("../src/model.jl")
include("../src/shapley_values.jl")

function train(model, d; numberofepochs=50, trainloader, onetrainloader, onetestloader)

    lossfunction(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
    opt = Flux.setup(Adam(1.0f-4), model)

    report(0, model, onetrainloader, onetestloader, lossfunction)

    for epoch in 1:numberofepochs
        for (x, y) in trainloader
            grads = Flux.gradient(model) do model
                ŷ = model(x)
                lossfunction(ŷ, y)
            end
            Flux.update!(opt, model, grads[1])
        end
        # if epoch % 10 == 0
        #     stats = report(epoch, model, onetrainloader, onetestloader, lossfunction)
        # end
        if epoch == numberofepochs
            stats = report(epoch, model, onetrainloader, onetestloader, lossfunction)
            push!(d["MODEL"], stats[end])
        end
    end

    return model, d
end

function compute_shapley_values(model, d, namesnetworks7, testtomask)
    r1 = [8, 12, 19, 85, 25, 21, 13, 36, 78, 43, 41, 84, 73, 15, 79, 6, 75]
    r2 = [9, 22, 47, 76, 52, 86, 70, 64, 56, 94, 72, 45, 96, 29]
    r3 = [92, 20, 24, 51, 44, 74, 31, 87, 48, 60, 35, 98, 65, 88, 57]
    r4 = [59, 89, 10, 63, 80, 28, 95, 32, 16, 4, 42, 77]
    r5 = [67, 39, 62, 3, 58]
    r6 =[49, 53, 100, 69, 61, 18, 83, 26, 93, 38, 82, 90, 2]
    r7 = [33, 66, 30, 81, 68, 40, 7, 17, 37, 11, 99, 97, 55, 23, 14, 1, 46, 5, 27, 54, 71, 50, 91, 34]
   


    networks7 = [r1, r2, r3, r4, r5, r6, r7]

    testtomaskdeepcopy = deepcopy(testtomask[1])
    for (i, name) in enumerate(namesnetworks7)
        shap = shaply_value_of_i(networks7[i], model, networks7, testtomask, testtomaskdeepcopy)
        println(shap)
        push!(d[name], shap)
    end
    return d
end


graphs, labels = load_schema_dataset(classification="AR")

d = Dict{String,Any}()
d["MODEL"] = []
namesnetworks7 = ("r1", "r2", "r3", "r4", "r5", "r6", "r7")
for i in 1:7
    d[namesnetworks7[i]] = []
end

testtomask = (graphs[:, :, :, :, 401:end], labels[:, 401:end])

for i in 1:15
    global d, graphs, labels, testtomask
    model = create_model(8, 1; classification="AR")
    model = model |> gpu
    

    trainloader = Flux.DataLoader((graphs[:, :, :, :, 1:400], labels[:, 1:400]), batchsize=1, shuffle=true)
    onetrainloader = Flux.DataLoader((graphs[:, :, :, :, 1:400], labels[:, 1:400]), batchsize=400, shuffle=true)
    onetestloader = Flux.DataLoader((graphs[:, :, :, :, 401:end], labels[:, 401:end]), batchsize=96, shuffle=true)

    #using gpu 
    trainloader = trainloader |> gpu
    onetrainloader = onetrainloader |> gpu
    onetestloader = onetestloader |> gpu
    model, d = train(model, d; numberofepochs=20, trainloader=trainloader, onetrainloader=onetrainloader, onetestloader=onetestloader)
 
    model = model |> cpu
    d = compute_shapley_values(model, d, namesnetworks7, testtomask)
end
jldsave("airport_restaurant_classification/data/shapleyvalues10random_15retraining.jld2"; d)

# sizes = [17, 14, 15,12,5,13,24]
# torem = []
# for i in 1:7
#      r =sample(setdiff(1:100,torem), sizes[i], replace=false)
#      println(r)
#      push!(torem, r...)
# end


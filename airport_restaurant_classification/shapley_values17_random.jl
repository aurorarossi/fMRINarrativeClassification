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

function create_random_parcellation()
    sizes = [7, 6, 6, 8, 6, 7, 9, 5, 2, 3, 6, 5, 5, 7, 10, 4, 4]
    torem = []
    networks17 = []
    for i in 1:17
         r =sample(setdiff(1:100,torem), sizes[i], replace=false)
         push!(networks17, r)
         push!(torem, r...)
    end
    return networks17
end

function compute_shapley_values(model, d,networks17, namesnetworks17, testtomask)
    
    
    testtomaskdeepcopy = deepcopy(testtomask[1])
    for (i, name) in enumerate(namesnetworks17)
        shap = approximate_shaply_value_of_i(i, model, networks17, testtomask, testtomaskdeepcopy,100)
        println(shap)
        push!(d[name], shap)
    end
    return d
end


graphs, labels = load_schema_dataset(classification="AR")
for p in 1:10
d = Dict{String,Any}()
d["MODEL"] = []
namesnetworks17 = ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "r16", "r17")

for i in 1:17
    d[namesnetworks17[i]] = []
end

testtomask = (graphs[:, :, :, :, 401:end], labels[:, 401:end])
networks17 = create_random_parcellation()

for i in 1:15
    #global d, graphs, labels, testtomask
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
    d = compute_shapley_values(model, d,networks17, namesnetworks17, testtomask)
end
jldsave("airport_restaurant_classification/data/shapleyvalues17random_15retraining_$(p).jld2"; d)
end

# sizes = [17, 14, 15,12,5,13,24]
# torem = []
# for i in 1:7
#      r =sample(setdiff(1:100,torem), sizes[i], replace=false)
#      println(r)
#      push!(torem, r...)
# end


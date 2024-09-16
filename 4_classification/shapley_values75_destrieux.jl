using CUDA, Statistics, JLD2
include("../utils.jl")
include("../src/model.jl")
include("../src/shapley_values.jl")

function train(model, d; numberofepochs=50, trainloader, onetrainloader, onetestloader)

    lossfunction(ŷ, y) = Flux.logitbinarycrossentropy(ŷ, y)
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

function compute_shapley_values(model, d, namesnetworks, testtomask)

    networks = [[i] for i in 1:75]

    testtomaskdeepcopy = deepcopy(testtomask[1])
    for (i, name) in enumerate(namesnetworks)
        shap = approximate_shaply_value_of_i(i, model, networks, testtomask, testtomaskdeepcopy, 75)
        println(shap)
        push!(d[name], shap)
    end
    return d
end

graphs, labels = load_schema_dataset_destrieux(classification="4C")

d = Dict{String,Any}()
d["MODEL"] = []
namesnetworks =[string(i) for i in 1:75]
for i in 1:75
    d[namesnetworks[i]] = []
end

testtomask = (graphs[:, :, :, :, 401:end], labels[:, 401:end])

for i in 1:15
    global d, graphs, labels, testtomask
    model = create_model_destrieux(8, 1; classification="4C")
    trainloader = Flux.DataLoader((graphs[:, :, :, :, 1:400], labels[:, 1:400]), batchsize=1, shuffle=true)
    onetrainloader = Flux.DataLoader((graphs[:, :, :, :, 1:400], labels[:, 1:400]), batchsize=400, shuffle=true)
    onetestloader = Flux.DataLoader((graphs[:, :, :, :, 401:end], labels[:, 401:end]), batchsize=96, shuffle=true)

    #using gpu 
    trainloader = trainloader |> gpu
    onetrainloader = onetrainloader |> gpu
    onetestloader = onetestloader |> gpu
    model = model |> gpu
    model, d = train(model, d; numberofepochs=20, trainloader=trainloader, onetrainloader=onetrainloader, onetestloader=onetestloader)
    model = model |> cpu
    d = compute_shapley_values(model, d, namesnetworks, testtomask)
end
jldsave("4_classification/data/shapleyvalues75_destrieux_15retraining.jld2"; d)


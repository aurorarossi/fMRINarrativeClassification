using CUDA, Statistics, Random, Flux, JLD2
include("utils.jl")
include("src/model.jl")

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

function permute_times(graphs)
    for i in 1:size(graphs,5)
        graphs[:, :, :, :, i] = graphs[:, :, randperm(8), :, i]
    end
    return graphs
end


graphs, labels = load_schema_dataset(classification="AR")

d = Dict{String,Any}()
d["MODEL"] = []
d["tp"] = []
d["tn"] = []
d["fp"] = []
d["fn"] = []



for i in 1:15
    global d, graphs, labels, testtomask
    model = create_model(8, 1; classification="AR")
    model = model |> gpu
    #permute time dimension 
    graphs = permute_times(graphs)
    trainloader = Flux.DataLoader((graphs[:, :, :, :, 1:400], labels[:, 1:400]), batchsize=1, shuffle=true)
    testloader = Flux.DataLoader((graphs[:, :, :, :, 401:end], labels[:, 401:end]), batchsize=1, shuffle=true)
    onetrainloader = Flux.DataLoader((graphs[:, :, :, :, 1:400], labels[:, 1:400]), batchsize=400, shuffle=true)
    onetestloader = Flux.DataLoader((graphs[:, :, :, :, 401:end], labels[:, 401:end]), batchsize=96, shuffle=true)

    #using gpu 
    trainloader = trainloader |> gpu
    onetrainloader = onetrainloader |> gpu
    onetestloader = onetestloader |> gpu

    model, d = train(model, d; numberofepochs=20, trainloader=trainloader, onetrainloader=onetrainloader, onetestloader=onetestloader)

    model = model |> cpu
    tp, tn, fp, fn = confusion_matrix(model, testloader)
    push!(d["tp"], tp)
    push!(d["tn"], tn)
    push!(d["fp"], fp)
    push!(d["fn"], fn)
end
JLD2.@save "airport_restaurant_classification/data/train_7_permutedtimes.jld2" d

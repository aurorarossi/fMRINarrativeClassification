include("utils.jl")

function readgraphs_MA!(g, graphs, labels, k, numberschema, numbersubject)
    for i in 0:3
        graphs[:, :, :, 1, k] = g[:, :, i+1]
        labels[:, k] = Flux.onehot(i % 2 == 0 ? 'M' : 'A', ['M', 'A'])
        k += 1
    end
    return graphs, labels, k
end

function readgraphs_AR!(g, graphs, labels, k, numberschema, numbersubject)
    datal = readdlm("files/airport_restaurant_labels/$(numbersubject)_schema-0$(numberschema)_events.txt")[1]
    for i in 0:3
        graphs[:, :, :, 1, k] = g[:, :, i+1]
        labels[:, k] = Flux.onehot(datal[i+1], ['A', 'R'])
        k += 1
    end
    return graphs, labels, k
end

function readgraphs_4C!(g, graphs, labels, k, numberschema, numbersubject)
    datal = readdlm("files/airport_restaurant_labels/$(numbersubject)_schema-0$(numberschema)_events.txt")[1]
    for i in 0:3
        graphs[:, :, :, 1, k] = g[:, :, i+1]
        labels[:, k] = Flux.onehot(datal[i+1] * (i % 2 == 0 ? 'M' : 'S'), ["AM", "AS", "RM", "RS"])
        k += 1
    end
    return graphs, labels, k
end


function load_schema_dataset(; classification)
    names = load_names_file()
    graphs = zeros(100, 100, 1, 1, 496)
    k = 1
    if classification == "4C"
        read = readgraphs_4C!
        numberlabels = 4
    elseif classification == "AR"
        read = readgraphs_AR!
        numberlabels = 2
    elseif classification == "MA"
        read = readgraphs_MA!
        numberlabels = 2
    end
    labels = zeros(numberlabels, 496)
    for (i, name) in enumerate(names)
        _, _, c = split(name, "/")
        g = npzread("data/static/$(c)_network_static.npy")[:, :, 1:4]
        graphs, labels, k = read(g, graphs, labels, k, c[25], c[1:7])
    end
    return graphs, labels
end

using CUDA, Statistics, Random, JLD2
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
        # if epoch % 5 == 0
        #      stats = report(epoch, model, onetrainloader, onetestloader, lossfunction)
        # end
        if epoch == numberofepochs
            stats = report(epoch, model, onetrainloader, onetestloader, lossfunction)
           push!(d["MODEL"], stats[end])
        end
    end

    return model, d
end




    graphs, labels = load_schema_dataset(classification="AR")

    d = Dict{String,Any}()
    d["MODEL"] = []

    for i in 1:15
    model = create_model(1, 8; classification="AR")
    model = model |> gpu

    
        global model, d, graphs, labels, testtomask
        #permute time dimension 
       # graphs = permute_times(graphs)
        trainloader = Flux.DataLoader((graphs[:, :, :, :, 1:400], labels[:, 1:400]), batchsize=1, shuffle=true)
        testloader = Flux.DataLoader((graphs[:, :, :, :, 401:end], labels[:, 401:end]), batchsize=1, shuffle=true)
        onetrainloader = Flux.DataLoader((graphs[:, :, :, :, 1:400], labels[:, 1:400]), batchsize=400, shuffle=true)
        onetestloader = Flux.DataLoader((graphs[:, :, :, :, 401:end], labels[:, 401:end]), batchsize=96, shuffle=true)

        #using gpu 
        trainloader = trainloader |> gpu
        onetrainloader = onetrainloader |> gpu
        onetestloader = onetestloader |> gpu
        Flux.reset!(model)

        model, d = train(model, d; numberofepochs=20, trainloader=trainloader, onetrainloader=onetrainloader, onetestloader=onetestloader)
        
    end
        JLD2.@save "/user/aurossi/home/fMRINarrativeClassification/static_FC/data/train_airportrestaurant.jld2" d
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

function compute_shapley_values(model, d, namesnetworks7, testtomask)
    vis = vcat(0:8, 50:57) .+ 1
    somot = vcat(9:14, 58:65) .+ 1
    dorsattn = vcat(15:22, 66:72) .+ 1
    ventattn = vcat(23:29, 73:77) .+ 1
    limbic = vcat(30:32, 78:79) .+ 1
    control = vcat(33:36, 80:88) .+ 1
    default = vcat(37:49, 89:99) .+ 1

    networks7 = [vis, somot, dorsattn, ventattn, limbic, control, default]

    testtomaskdeepcopy = deepcopy(testtomask[1])
    for (i, name) in enumerate(namesnetworks7)
        shap = shaply_value_of_i(networks7[i], model, networks7, testtomask, testtomaskdeepcopy)
        println(shap)
        push!(d[name], shap)
    end
    return d
end


graphs, labels = load_schema_dataset(classification="AR")
graphs = (graphs .- mean(graphs))./ std(graphs)

d = Dict{String,Any}()
d["MODEL"] = []
namesnetworks7 = ("vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default")
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
jldsave("airport_restaurant_classification/data/shapleyvalues7_15retraining_normalized.jld2"; d)

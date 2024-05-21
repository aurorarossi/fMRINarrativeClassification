using CUDA, Statistics, JLD2
include("utils.jl")
include("src/model.jl")
include("src/shapley_values.jl")

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

function compute_shapley_values(model, d, namesnetworks17, testtomask)
    visualA = vcat(0:3,50:52).+1
    visualB = vcat(4:6,53:55).+1
    somatomotorA = vcat(7:8,56:59).+1
    somatomotorB = vcat(9:12,60:63).+1
    dorsalattentionA = vcat(13:15,64:66).+1
    dorsalattentionB = vcat(16:19,67:69).+1
    ventralattentionA = vcat(20:24,70:73).+1
    ventralattentionB = vcat(25:26,74:76).+1
    limbicA = vcat(27,77).+1
    limbicB = vcat(28:29,78).+1
    controlA = vcat(30:32,79:81).+1
    controlB = vcat(33,82:85).+1
    controlC = vcat(34:36,86:87).+1
    defaultA = vcat(37:39,88:91).+1
    defaultB = vcat(40:46,92:94).+1
    defaultC = vcat(47:48,95:96).+1
    temporalparietal = vcat(49,97:99).+1

    networks17 = [visualA, visualB, somatomotorA, somatomotorB, dorsalattentionA, dorsalattentionB, ventralattentionA, ventralattentionB, limbicA, limbicB, controlA, controlB, controlC, defaultA, defaultB, defaultC, temporalparietal]

    testtomaskdeepcopy = deepcopy(testtomask[1])
    for (i, name) in enumerate(namesnetworks17)
        shap = approximate_shaply_value_of_i(i, model, networks17, testtomask, testtomaskdeepcopy,100)
        println(shap)
        push!(d[name], shap)
    end
    return d
end

graphs, labels = load_schema_dataset(classification="4C")

d = Dict{String,Any}()
d["MODEL"] = []
namesnetworks17 =( "visualA", "visualB", "somatomotorA", "somatomotorB", "dorsalattentionA", "dorsalattentionB", "ventralattentionA", "ventralattentionB", "limbicA", "limbicB", "controlA", "controlB", "controlC", "defaultA", "defaultB", "defaultC", "temporalparietal")
for i in 1:17
    d[namesnetworks17[i]] = []
end

testtomask = (graphs[:, :, :, :, 401:end], labels[:, 401:end])

for i in 1:15
    global d, graphs, labels, testtomask
    model = create_model(4, 5; classification="4C")
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
    d = compute_shapley_values(model, d, namesnetworks17, testtomask)
end
jldsave("4_classification/data/shapleyvalues17_15retraining.jld2"; d)



using CUDA, Statistics, Random
include("../utils.jl")
include("../src/model.jl")

function train(model, d; numberofepochs=50, trainloader, onetrainloader, onetestloader)

    lossfunction(ŷ, y) = Flux.logitbinarycrossentropy(ŷ, y)
    opt = Flux.setup(Adam(1.0f-4), model)

    report_inter_intra(0, model, onetrainloader, onetestloader, lossfunction)

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
            acc_intra, std_acc_intra, std_acc_inter, totalstd = report_inter_intra(epoch, model, onetrainloader, onetestloader, lossfunction)
            push!(d["acc_intra"],acc_intra)
            push!(d["std_acc_intra"],std_acc_intra)
            push!(d["std_acc_inter"],std_acc_inter)
            push!(d["totalstd"],totalstd)
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
    d["acc_intra"] =[]
    d["std_acc_intra"] = []
    d["std_acc_inter"] = []
    d["totalstd"] = []

    for i in 1:15
    model = create_model(8, 1; classification="AR")
    model = model |> gpu

    
        global model, d, graphs, labels, testtomask
        #permute time dimension 
       # graphs = permute_times(graphs)
        trainset,testset,trainlabel,testlabel = select_subjects(graphs,labels,2)
        trainloader = Flux.DataLoader((trainset[:, :, :, :, 1:400], trainlabel[:, 1:400]), batchsize=1, shuffle=true)
        onetrainloader = Flux.DataLoader((trainset[:, :, :, :, 1:400], trainlabel[:, 1:400]), batchsize=400, shuffle=true)
        onetestloader = Flux.DataLoader((testset, testlabel), batchsize=96, shuffle=false)



        #using gpu 
        trainloader = trainloader |> gpu
        onetrainloader = onetrainloader |> gpu
        onetestloader = onetestloader |> gpu
        Flux.reset!(model)

        model, d = train(model, d; numberofepochs=20, trainloader=trainloader, onetrainloader=onetrainloader, onetestloader=onetestloader)
        
        #model = model |> cpu
        # tp, tn, fp, fn = confusion_matrix(model, testloader)
        # push!(d["tp"], tp)
        # push!(d["tn"], tn)
        # push!(d["fp"], fp)
        # push!(d["fn"], fn)

    end
    
    JLD2.@save "airport_restaurant_classification/data/intra_inter_15retraining.jld2" d
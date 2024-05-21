using DelimitedFiles, Flux, NPZ

function load_names_file()
    name1 = readdlm("files/to_process/to_process_schema1.txt")
    name2 = readdlm("files/to_process/to_process_schema2.txt")
    name3 = readdlm("files/to_process/to_process_schema3.txt")
    name4 = readdlm("files/to_process/to_process_schema4.txt")
    return vcat(name1, name2, name3, name4)
end

function readgraphs_4C!(g, graphs, labels, k, numberschema, numbersubject)
    datal = readdlm("files/airport_restaurant_labels/$(numbersubject)_schema-0$(numberschema)_events.txt")[1]
    for i in 0:3
        graphs[:, :, :, 1, k] = g[:, :, (8*i+1):(8*(i+1))]
        labels[:, k] = Flux.onehot(datal[i+1] * (i % 2 == 0 ? 'M' : 'S'), ["AM", "AS", "RM", "RS"])
        k += 1
    end
    return graphs, labels, k
end

function readgraphs_AR!(g, graphs, labels, k, numberschema, numbersubject)
    datal = readdlm("files/airport_restaurant_labels/$(numbersubject)_schema-0$(numberschema)_events.txt")[1]
    for i in 0:3
        graphs[:, :, :, 1, k] = g[:, :, (8*i+1):(8*(i+1))]
        labels[:, k] = Flux.onehot(datal[i+1], ['A', 'R'])
        k += 1
    end
    return graphs, labels, k
end

function readgraphs_MA!(g, graphs, labels, k, numberschema, numbersubject)
    for i in 0:3
        graphs[:, :, :, 1, k] = g[:, :, (8*i+1):(8*(i+1))]
        labels[:, k] = Flux.onehot(i % 2 == 0 ? 'M' : 'A', ['M', 'A'])
        k += 1
    end
    return graphs, labels, k
end

function load_schema_dataset(; classification)
    names = load_names_file()
    graphs = zeros(100, 100, 8, 1, 496)
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
        g = npzread("data/networks/$(c)_network_7networks.npy")[:, :, 1:32]
        graphs, labels, k = read(g, graphs, labels, k, c[25], c[1:7])
    end
    return graphs, labels
end

function eval_accuracy(model, data_loader)
    (x, y) = data_loader
    ŷ = model(x)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    return acc
end

function eval_loss_accuracy(model, data_loader, lossfunction)
    (x, y) = first(data_loader)
    ŷ = model(x)
    loss = lossfunction(ŷ, y)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    return (loss=loss, acc=acc)
end

function report(epoch, model, train_loader, test_loader, lossfunction)
    train_loss, train_acc = eval_loss_accuracy(model, train_loader, lossfunction)
    test_loss, test_acc = eval_loss_accuracy(model, test_loader, lossfunction)
    println("Epoch: $epoch  $((; train_loss, train_acc))  $((; test_loss, test_acc))")
    return (train_loss, train_acc, test_loss, test_acc)
end

function confusion_matrix(model, data_loader)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for (x,y) in data_loader
        ŷ = model(x)
        ŷ = Flux.onecold(ŷ)
        y = Flux.onecold(y)
        tp += sum((y .== 1) .& (ŷ .== 1))
        tn += sum((y .== 2) .& (ŷ .== 2))
        fp += sum((y .== 2) .& (ŷ .== 1))
        fn += sum((y .== 1) .& (ŷ .== 2))
    end
    return tp, tn, fp, fn
end

function confusion_matrix_4classification(model, data_loader)
    tp = zeros(Int, 4)
    tn = zeros(Int, 4)
    fp = zeros(Int, 4)
    fn = zeros(Int, 4)
    for (x,y) in data_loader
        ŷ = model(x)
        ŷ = Flux.onecold(ŷ)
        y = Flux.onecold(y)
        for i in 1:4
            tp[i] += sum((y .== i) .& (ŷ .== i))
            tn[i] += sum((y .!= i) .& (ŷ .!= i))
            fp[i] += sum((y .!= i) .& (ŷ .== i))
            fn[i] += sum((y .== i) .& (ŷ .!= i))
        end
    end
    return tp, tn, fp, fn
end

function precision(tp, fp)
    return tp / (tp + fp)
end

function recall(tp, fn)
    return tp / (tp + fn)
end

function f1_score(precision, recall)
    return 2 * precision * recall / (precision + recall)
end


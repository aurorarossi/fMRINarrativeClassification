using DelimitedFiles, Flux, NPZ, StatsBase

function load_names_file()
    name1 = readdlm("files/to_process/to_process_schema1.txt")
    name2 = readdlm("files/to_process/to_process_schema2.txt")
    name3 = readdlm("files/to_process/to_process_schema3.txt")
    name4 = readdlm("files/to_process/to_process_schema4.txt")
    return vcat(name1, name2, name3, name4)
end

function select_subjects(graphs, labels, neurons)
    subjects = sample(1:31, 6, replace=false)
    println(subjects)
    subjecttrain = setdiff(1:31,subjects)
    testset = zeros(100, 100, 8, 1, 96)
    testlabel = zeros(neurons,96)
    trainset = zeros(100, 100, 8, 1, 400)
    trainlabel = zeros(neurons,400)
    m=1
    
    for i in subjecttrain
        trainset[:, :, :, 1, m:m+15] = graphs[:, :, :, 1, (i-1)*16+1:(i-1)*16+16]
        trainlabel[:, m:m+15] = labels[:, (i-1)*16+1:(i-1)*16+16]
        m += 16
    end
    testset = zeros(100, 100, 8, 1, 96)
    k=1
    for i in subjects
        testset[:, :, :, 1, k:k+15] = graphs[:, :, :, 1, (i-1)*16+1:(i-1)*16+16]
        testlabel[:, k:k+15] = labels[:, (i-1)*16+1:(i-1)*16+16]
        k += 16
    end
    return trainset, testset, trainlabel, testlabel
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

function load_schema_dataset17(; classification)
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
        g = npzread("data/networks/$(c)_network.npy")[:, :, 1:32]
        graphs, labels, k = read(g, graphs, labels, k, c[25], c[1:7])
    end
    return graphs, labels
end

function load_schema_dataset_desikan(; classification)
    names = load_names_file()
    graphs = zeros(70, 70, 8, 1, 496)
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
        g = npzread("data/desikan/$(c)_network_desikan.npy")[:, :, 1:32]
        graphs, labels, k = read(g, graphs, labels, k, c[25], c[1:7])
    end
    return graphs, labels
end

function load_schema_dataset_destrieux(; classification)
    names = load_names_file()
    graphs = zeros(75, 75, 8, 1, 496)
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
        g = npzread("data/destrieux/$(c)_network_destrieux.npy")[:, :, 1:32]
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


function eval_loss_accuracy_inter_intra_subject(model, data_loader, lossfunction)
    (x, y) = first(data_loader)
    ŷ = model(x)
    loss = lossfunction(ŷ, y)
    
    num_subjects = 6
    data_points_per_subject = 16
    
    accuracies_subjects = Float64[]
    std_subjects = Float64[]
    
    for i in 1:num_subjects
        start_idx = (i - 1) * data_points_per_subject + 1
        end_idx = i * data_points_per_subject
        subject_ŷ = ŷ[:, start_idx:end_idx]
        subject_y = y[:, start_idx:end_idx]
        subject_acc = mean(Flux.onecold(subject_ŷ) .== Flux.onecold(subject_y))
        subject_std = std(Flux.onecold(subject_ŷ) .== Flux.onecold(subject_y))
        push!(accuracies_subjects, subject_acc)
        push!(std_subjects, subject_std)
    end
    
    avg_acc_intra = round(100 * mean(accuracies_subjects); digits=2)
    std_acc_intra = round(100 * mean(std_subjects); digits=2)
    std_acc_inter = round(100 * std(accuracies_subjects); digits=2)
    total_std = std(Flux.onecold(ŷ) .== Flux.onecold(y))
    
    return (loss=loss, acc_intra=avg_acc_intra, std_acc_intra=std_acc_intra, std_acc_inter=std_acc_inter, total_std=total_std)
end

function report_inter_intra(epoch, model,train_loader, test_loader, lossfunction)
    loss, acc_intra, std_acc_intra, std_acc_inter, total_std = eval_loss_accuracy_inter_intra_subject(model, test_loader, lossfunction)
    println("Epoch: $epoch $((; loss, acc_intra, std_acc_intra, std_acc_inter))")
    return  acc_intra, std_acc_intra, std_acc_inter, total_std
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


using Flux

function create_model(t1, t2; classification)
    if classification == "4C"
        numberlabels = 4
    elseif classification == "AR"
        numberlabels = 2
    elseif classification == "MA"
        numberlabels = 2
    end
    model = Chain(
        Conv((100, 100, t1), 1 => 128, relu; init=Flux.kaiming_uniform),
        MaxPool((1, 1, t2)),
        Flux.flatten,
        Dense(128, 64, relu; init=Flux.kaiming_uniform),
        Dense(64, 32, relu; init=Flux.kaiming_uniform),
        Dense(32, numberlabels)
    )
    return model
end


function create_model_desikan(t1, t2; classification)
    if classification == "4C"
        numberlabels = 4
    elseif classification == "AR"
        numberlabels = 2
    elseif classification == "MA"
        numberlabels = 2
    end
    model = Chain(
        Conv((70, 70, t1), 1 => 128, relu; init=Flux.kaiming_uniform),
        MaxPool((1, 1, t2)),
        Flux.flatten,
        Dense(128, 64, relu; init=Flux.kaiming_uniform),
        Dense(64, 32, relu; init=Flux.kaiming_uniform),
        Dense(32, numberlabels)
    )
    return model
end

function create_model_destrieux(t1, t2; classification)
    if classification == "4C"
        numberlabels = 4
    elseif classification == "AR"
        numberlabels = 2
    elseif classification == "MA"
        numberlabels = 2
    end
    model = Chain(
        Conv((75, 75, t1), 1 => 128, relu; init=Flux.kaiming_uniform),
        MaxPool((1, 1, t2)),
        Flux.flatten,
        Dense(128, 64, relu; init=Flux.kaiming_uniform),
        Dense(64, 32, relu; init=Flux.kaiming_uniform),
        Dense(32, numberlabels)
    )
    return model
end
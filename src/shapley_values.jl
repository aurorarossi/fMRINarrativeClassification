using Combinatorics
include("../utils.jl")

function mask(mygraph, maskedregions)
    maskedregions = setdiff(collect(1:100), maskedregions)
    mygraph[maskedregions, :, :, :] .= 0
    mygraph[:, maskedregions, :, :] .= 0
    return mygraph
end

function evaluate_model!(subset, testtomask, model, testtomaskdeepcopy1)
    subset = vcat(subset...)
    testtomaskdeepcopy1 .= testtomask[1]
    for i in 1:96
        testtomaskdeepcopy1[:, :, :, :, i] = mask(testtomaskdeepcopy1[:, :, :, :, i], subset)
    end
    return eval_accuracy(model, (testtomaskdeepcopy1, testtomask[2]))
end

function shaply_value_of_i(i, model, networks, testtomask, testtomaskdeepcopy)
    n = length(networks)
    s = 0
    for subset in combinations(setdiff(networks, [i]))
        s += binomial(n - 1, n - length(subset) - 1)^-1 * (evaluate_model!(subset ∪ i, testtomask, model, testtomaskdeepcopy) - evaluate_model!(subset, testtomask, model, testtomaskdeepcopy))
    end
    return s / n
end

function approximate_shaply_value_of_i(i, model, networks, testtomask, testtomaskdeepcopy, samplesize)
    s = 0
    area = networks[i]
    for _ in 1:samplesize
        p = randperm(length(networks))
        subset = networks[p[1:findfirst(x -> x == i, p)-1]]
        s += (evaluate_model!(subset ∪ area, testtomask, model, testtomaskdeepcopy) - evaluate_model!(subset, testtomask, model, testtomaskdeepcopy))
    end
    return s / samplesize
end

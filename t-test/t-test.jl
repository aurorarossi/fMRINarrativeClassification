using JLD2, HypothesisTests, Statistics

function t_test7(data)
    data = data["d"]
    networks = ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"] 

    mean_values = Dict()
    for key in ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"]
        mean_values[key] = mean(data[key])
    end

    std_values = Dict()
    for key in ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"]
        std_values[key] = std(data[key])
    end

    function total_mean_except_region(data)
        values = []
        for key in ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"]
            push!(values, data[key]...)

        end
        return mean(values), std(values)
    end


    significance = zeros(17)
        i=1
    for key in ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"]
        total_mean = total_mean_except_region(data)
        println(key)
        display(pvalue(OneSampleTTest(mean_values[key], std_values[key], 15, total_mean[1]); tail = :right))
        println(pvalue(OneSampleTTest(mean_values[key], std_values[key], 15, total_mean[1]); tail = :right)<0.01)
        significance[i] = Int(pvalue(OneSampleTTest(mean_values[key], std_values[key], 15, total_mean[1]); tail = :right)<0.01)
        i+=1
        println()
    end
    return significance
end

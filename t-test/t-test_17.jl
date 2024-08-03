using JLD2, HypothesisTests, Statistics

#data = load("airport_restaurant_classification/data/shapleyvalues17_15retraining_500sample.jld2")
function t_test17(data)
    data = data["d"]
    networks = ["visualA", "visualB", "somatomotorA", "somatomotorB", "dorsalattentionA", "dorsalattentionB", "ventralattentionA", "ventralattentionB", "limbicA", "limbicB", "controlA", "controlB", "controlC", "defaultA", "defaultB", "defaultC", "temporalparietal"] 

    mean_values = Dict()
    for key in ["visualA", "visualB", "somatomotorA", "somatomotorB", "dorsalattentionA", "dorsalattentionB", "ventralattentionA", "ventralattentionB", "limbicA", "limbicB", "controlA", "controlB", "controlC", "defaultA", "defaultB", "defaultC", "temporalparietal"]
        mean_values[key] = mean(data[key])
    end

    std_values = Dict()
    for key in ["visualA", "visualB", "somatomotorA", "somatomotorB", "dorsalattentionA", "dorsalattentionB", "ventralattentionA", "ventralattentionB", "limbicA", "limbicB", "controlA", "controlB", "controlC", "defaultA", "defaultB", "defaultC", "temporalparietal"]
        std_values[key] = std(data[key])
    end

    function total_mean_except_region(data)
        values = []
        for key in ["visualA", "visualB", "somatomotorA", "somatomotorB", "dorsalattentionA", "dorsalattentionB", "ventralattentionA", "ventralattentionB", "limbicA", "limbicB", "controlA", "controlB", "controlC", "defaultA", "defaultB", "defaultC", "temporalparietal"]
            push!(values, data[key]...)
        end
        return mean(values), std(values)
    end

    significance = zeros(17)
    i=1
    for key in ["visualA", "visualB", "somatomotorA", "somatomotorB", "dorsalattentionA", "dorsalattentionB", "ventralattentionA", "ventralattentionB", "limbicA", "limbicB", "controlA", "controlB", "controlC", "defaultA", "defaultB", "defaultC", "temporalparietal"]
        total_mean = total_mean_except_region(data)
        println(key)
        display(pvalue(OneSampleTTest(mean_values[key], std_values[key], 15, total_mean[1]); tail = :right))
        println(pvalue(OneSampleTTest(mean_values[key], std_values[key], 15, total_mean[1]); tail = :right)<0.01)
        significance[i] = Int.(pvalue(OneSampleTTest(mean_values[key], std_values[key], 15, total_mean[1]); tail = :right)<0.05)
        i+=1
        println()
    end
return significance
end
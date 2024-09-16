using NPZ, Statistics, DelimitedFiles

names = readdlm("files/to_process/to_process_schema3.txt")
for l in 1:length(names)
    println(names[l])
    name = names[l]
    _,_,c = split(name,"/")
    timeseries = npzread("timeseries/schema-run3-100/$(c)_timeseries_destrieux.npy")

    T = 20
    num_regions = 75
    global k=1
    window = collect(1:15:(size(timeseries,2)-20))
    network = zeros(num_regions,num_regions,length(window))
   

    for t in window
        global k
        for i in 1:num_regions
            for j in 1:num_regions
                if i == 42 || j == 42
                    continue
                else
                    network[i,j,k]=cor(timeseries[i,t:t+T], timeseries[j,t:t+T])
                end
            end
        end
        k+=1
    end

    npzwrite("data/destrieux/$(c)_network_destrieux.npy", network)
end
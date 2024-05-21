using NPZ, Statistics, DelimitedFiles

names = readdlm("to_process_schema4.txt")
for l in 1:length(names)
    println(names[l])
    name = names[l]
    _,_,c = split(name,"/")
    timeseries = npzread("timeseries/schema-run4-100/$(c)_timeseries.npy")

    T = 20
    num_regions = 100

    global k=1
    window = collect(1:15:(size(timeseries,2)-20))
    network = zeros(num_regions,num_regions,length(window))
   

    for t in window
        global k
        for i in 1:num_regions
            for j in 1:num_regions
                network[i,j,k]=cor(timeseries[i,t:t+T], timeseries[j,t:t+T])
            end
        end
        k+=1
    end

    npzwrite("schema/$(c)_network.npy", network)
end
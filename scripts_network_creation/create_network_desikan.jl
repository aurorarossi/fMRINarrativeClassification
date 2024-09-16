using NPZ, Statistics, DelimitedFiles

names = readdlm("files/to_process/to_process_schema1.txt")
for l in 1:length(names)
    println(names[l])
    name = names[l]
    _,_,c = split(name,"/")
    timeseries = npzread("timeseries/schema-run1-100/$(c)_timeseries_desikan.npy")

    T = 20
    num_regions = 70
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

    npzwrite("data/desikan/$(c)_network_desikan.npy", network)
end
using NPZ, Statistics, DelimitedFiles

names = readdlm("files/to_process/to_process_schema3.txt")
for l in 1:length(names)
    println(names[l])
    name = names[l]
    _,_,c = split(name,"/")
    timeseries = npzread("timeseries/schema-run3-100/$(c)_timeseries.npy")

    T = 20
    num_regions = 100

    global k=1
    window = collect(1:15:(size(timeseries,2)-20))
    network = zeros(num_regions,num_regions,4)
   

    for t in window[1:9:end][1:4]
        global k
        for i in 1:num_regions
            for j in 1:num_regions
                T = t + 135
                if t+T > size(timeseries,2)
                    T = size(timeseries,2)
                end
                network[i,j,k]=cor(timeseries[i,t:T], timeseries[j,t:T])
            end
        end
        k+=1
    end

    npzwrite("data/static/$(c)_network_static.npy", network)
end
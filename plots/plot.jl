using Random, CairoMakie, Statistics, LaTeXStrings

function shapvalues_plot(data,title,format)
    CairoMakie.activate!(type = format)
    data = data["d"]
    networks = ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"] 
    networks_real_names = [L"Visual$$", L"Somatomotor$$", L"""Dorsal Attention$$""", L"""Ventral Attention$$""", L"Limbic$$", L"Control$$", L"Default$$"]

    mean_values = Dict()
    for key in ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"]
        mean_values[key] = mean(data[key])
    end

    std_values = Dict()
    for key in ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"]
        std_values[key] = std(data[key])
    end
    shapvalues = [mean_values["vis"], mean_values["somot"], mean_values["dorsattn"], mean_values["ventattn"], mean_values["limbic"], mean_values["control"], mean_values["default"]]

    std_values = [std_values["vis"], std_values["somot"], std_values["dorsattn"], std_values["ventattn"], std_values["limbic"], std_values["control"], std_values["default"]]

    permsort = sortperm(shapvalues, rev = true)
    color = categorical_colors(:RdBu_4, 4)
    fontsize_theme = Theme(fontsize=20)
set_theme!(fontsize_theme)
    p = barplot(shapvalues[permsort],   
    axis = (xticks = (1:7, networks_real_names[permsort]),yticks =(0:5:15, [L"0",L"5",L"10",L"15"]) ,xticklabelrotation=pi/9,xlabel = L"Subnetwork$$", ylabel = L"Shapley Value$$", title = title) ,legend = false, color = color[end] )
    errorbars!(collect(1:7),shapvalues[permsort], std_values,  whiskerwidth = 10, color = :black)

    display(p)

    return p
end

function shapvalues17_plot(data,title,format)
    CairoMakie.activate!(type = format)
    data = data["d"]
    networks = ["visualA", "visualB", "somatomotorA", "somatomotorB", "dorsalattentionA", "dorsalattentionB", "ventralattentionA", "ventralattentionB", "limbicA", "limbicB", "controlA", "controlB", "controlC", "defaultA", "defaultB", "defaultC", "temporalparietal"]
    networks_real_names = [L"Visual A$$", L"Visual B$$", L"Somatomotor A$$", L"Somatomotor B$$", L"""Dorsal Attention A$$""", L"""Dorsal Attention B$$""", L"""Ventral Attention A$$""", L"""Ventral Attention B$$""", L"Limbic A$$", L"Limbic B$$", L"Control A$$", L"Control B$$", L"Control C$$", L"Default A$$", L"Default B$$", L"Default C$$", L"Temporal Parietal$$"]

    mean_values = Dict()
    for key in networks
        mean_values[key] = mean(data[key])
    end

    std_values = Dict()
    for key in networks
        std_values[key] = std(data[key])
    end
    shapvalues = [mean_values["visualA"], mean_values["visualB"], mean_values["somatomotorA"], mean_values["somatomotorB"], mean_values["dorsalattentionA"], mean_values["dorsalattentionB"], mean_values["ventralattentionA"], mean_values["ventralattentionB"], mean_values["limbicA"], mean_values["limbicB"], mean_values["controlA"], mean_values["controlB"], mean_values["controlC"], mean_values["defaultA"], mean_values["defaultB"], mean_values["defaultC"], mean_values["temporalparietal"]]

    std_values = [std_values["visualA"], std_values["visualB"], std_values["somatomotorA"], std_values["somatomotorB"], std_values["dorsalattentionA"], std_values["dorsalattentionB"], std_values["ventralattentionA"], std_values["ventralattentionB"], std_values["limbicA"], std_values["limbicB"], std_values["controlA"], std_values["controlB"], std_values["controlC"], std_values["defaultA"], std_values["defaultB"], std_values["defaultC"], std_values["temporalparietal"]]

    permsort = sortperm(shapvalues, rev = true)
    color = categorical_colors(:RdBu_4, 4)
    fontsize_theme = Theme(fontsize=20)
    set_theme!(fontsize_theme)
    p = barplot(shapvalues[permsort],   
    axis = (xticks = (1:17, networks_real_names[permsort]),yticks =(0:1:6, [ L"0",L"1",L"2",L"3",L"4",L"5",L"6"]) ,xticklabelrotation=pi/5,xlabel = L"Subnetwork$$", ylabel = L"Shapley Value$$", title = title) ,legend = false, color = color[end] )
    errorbars!(collect(1:17),shapvalues[permsort], std_values,  whiskerwidth = 10, color = :black)

    display(p)

    return p
end

function shapvalues100_plot(data,title,format)
    vis = vcat(0:8, 50:57) .+ 1
    somot = vcat(9:14, 58:65) .+ 1
    dorsattn = vcat(15:22, 66:72) .+ 1
    ventattn = vcat(23:29, 73:77) .+ 1
    limbic = vcat(30:32, 78:79) .+ 1
    control = vcat(33:36, 80:88) .+ 1
    default = vcat(37:49, 89:99) .+ 1

    palette = reverse(categorical_colors(:Paired_7, 7))

    region_to_subnetwork = fill(:unknown, 100)
region_to_subnetwork[vis] .= :vis
region_to_subnetwork[somot] .= :somot
region_to_subnetwork[dorsattn] .= :dorsattn
region_to_subnetwork[ventattn] .= :ventattn
region_to_subnetwork[limbic] .= :limbic
region_to_subnetwork[control] .= :control
region_to_subnetwork[default] .= :default

subnetwork_colors = Dict(
    :vis => palette[2],
    :somot => palette[1],
    :dorsattn => palette[3],
    :ventattn => palette[4],
    :limbic => palette[5],
    :control => palette[7],
    :default => palette[6],
    :unknown => :gray
)


    CairoMakie.activate!(type = format)
    data = data["d"]
    networks = [string(i) for i in 1:100]
    networks_real_names = [string(i) for i in 1:100]

    mean_values = Dict()
    for key in networks
        mean_values[key] = mean(data[key])
    end

    std_values = Dict()
    for key in networks
        std_values[key] = std(data[key])
    end
    shapvalues = [mean_values[string(i)] for i in 1:100]

    std_values = [std_values[string(i)] for i in 1:100]

    permsort = sortperm(shapvalues, rev = true)

    region_colors = [subnetwork_colors[region_to_subnetwork[i]] for i in permsort]
    fontsize_theme = Theme(fontsize=20)
    set_theme!(fontsize_theme)
    fig = Figure(size = (1400, 600))
    
    # Create a subplot for the main plot
    ax = Axis(fig[1, 1], title = title, xticks = (1:100, networks_real_names[permsort]), 
              yticks = (0:1:6, [L"0", L"1", L"2", L"3", L"4", L"5", L"6"]),
              xticklabelrotation = pi/2, xticklabelsize = 12, xlabel = L"Schaefer brain regions$$", ylabel = L"Shapley Value$$")
    
    barplot!(ax, shapvalues[permsort], color = region_colors)
    errorbars!(ax, collect(1:100), shapvalues[permsort], std_values, whiskerwidth = 10, color = :black)

    # Create a subplot for the scatter plot and text labels
    ax_legend = Axis(fig[1, 2], xautolimitmargin = (0, 0), yautolimitmargin = (0.1, 0.1), tellwidth = true, width=350)
    hidedecorations!(ax_legend)
    hidespines!(ax_legend)

    legend_labels = [L"Visual$$", L"Somatomotor$$", L"Dorsal Attention$$", L"Ventral Attention$$", L"Limbic$$", L"Control$$", L"Default$$"]
    legend_colors = [subnetwork_colors[:vis], subnetwork_colors[:somot], subnetwork_colors[:dorsattn], subnetwork_colors[:ventattn], subnetwork_colors[:limbic], subnetwork_colors[:control], subnetwork_colors[:default]]

        for (i, label) in enumerate(legend_labels)
        scatter!(ax_legend, [0], [-i], color = legend_colors[i], markersize = 15)  # Reduce marker size
        text!(ax_legend, [0], [-i], text = label, align=(:left,:center), offset = (10,0) )  # Adjust label position
    end

    display(fig)
    display(fig)




    return fig
end

function shapvaluesrandom_plot(data,title,format)
    CairoMakie.activate!(type = format)
    data = data["d"]
    networks = ["r1","r2","r3","r4","r5" , "r6", "r7"] 
    networks_real_names = [L"Random 1$$", L"Random 2$$", L"Random 3$$", L"Random 4$$", L"Random 5$$", L"Random 6$$", L"Random 7$$"]

    mean_values = Dict()
    for key in ["r1","r2","r3","r4","r5" , "r6", "r7"]
        mean_values[key] = mean(data[key])
    end

    std_values = Dict()
    for key in ["r1","r2","r3","r4","r5" , "r6", "r7"]
        std_values[key] = std(data[key])
    end
    shapvalues = [mean_values["r1"], mean_values["r2"], mean_values["r3"], mean_values["r4"], mean_values["r5"], mean_values["r6"], mean_values["r7"]]

    std_values = [std_values["r1"], std_values["r2"], std_values["r3"], std_values["r4"], std_values["r5"], std_values["r6"], std_values["r7"]]

    permsort = sortperm(shapvalues, rev = true)
    color = categorical_colors(:RdBu_4, 4)
    fontsize_theme = Theme(fontsize=20)
set_theme!(fontsize_theme)
    p = barplot(shapvalues[permsort],   
    axis = (xticks = (1:7, networks_real_names[permsort]),yticks =(0:5:15, [L"0",L"5",L"10",L"15"]) ,xticklabelrotation=pi/9,xlabel = L"Random Subnetwork$$", ylabel = L"Shapley Value$$", title = title) ,legend = false, color = color[end] )
    errorbars!(collect(1:7),shapvalues[permsort], std_values,  whiskerwidth = 10, color = :black)

    display(p)

    return p
end


function shapvalues_scatterplot(data,title,format)
    CairoMakie.activate!(type = format)
    data = data["d"]
    networks = ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"] 
    networks_real_names = [L"Visual$$", L"Somatomotor$$", L"""Dorsal Attention$$""", L"""Ventral Attention$$""", L"Limbic$$", L"Control$$", L"Default$$"]

    mean_values = Dict()
    for key in ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"]
        mean_values[key] = mean(data[key])
    end

    std_values = Dict()
    for key in ["vis", "somot", "dorsattn", "ventattn", "limbic", "control", "default"]
        std_values[key] = std(data[key])
    end
    shapvalues = [data["vis"], data["somot"], data["dorsattn"], data["ventattn"], data["limbic"], data["control"], data["default"]]

    shapvaluesmean = [mean_values["vis"], mean_values["somot"], mean_values["dorsattn"], mean_values["ventattn"], mean_values["limbic"], mean_values["control"], mean_values["default"]]
    


    std_values = [std_values["vis"], std_values["somot"], std_values["dorsattn"], std_values["ventattn"], std_values["limbic"], std_values["control"], std_values["default"]]

    permsort = sortperm(shapvaluesmean, rev = true)
    color = categorical_colors(:RdBu_4, 4)
    fontsize_theme = Theme(fontsize=20)
    set_theme!(fontsize_theme)
    x = []
    for i in 1:7
        push!(x, ones(size(shapvalues[permsort[i]])) * i)
    end
    x = vcat(x...)
    y = Float32.(vcat(shapvalues[permsort]...))
    p = scatter(x,y, axis = (xticks = (1:7, networks_real_names[permsort]),yticks =(0:5:15, [L"0",L"5",L"10",L"15"]) ,xticklabelrotation=pi/9,xlabel = L"Subnetwork$$", ylabel = L"Shapley Value$$", title = title) ,legend = false, color = color[end] )
    scatter!(1:7, shapvaluesmean[permsort], color = :red, markersize = 20, marker=:hline, label = "Mean Values")
   # errorbars!(collect(1:7),shapvalues[permsort], std_values,  whiskerwidth = 10, color = :black)
    println(shapvaluesmean[permsort])
    display(p)

    return p
end

function shapvaluesdesikan_plot(data,title,format)
 


    CairoMakie.activate!(type = format)
    data = data["d"]
    networks = [string(i) for i in 1:70]
    networks_real_names = [string(i) for i in 1:70]

    mean_values = Dict()
    for key in networks
        mean_values[key] = mean(data[key])
    end

    std_values = Dict()
    for key in networks
        std_values[key] = std(data[key])
    end
    shapvalues = [mean_values[string(i)] for i in 1:70]

    std_values = [std_values[string(i)] for i in 1:70]

    permsort = sortperm(shapvalues, rev = true)

    color = categorical_colors(:RdBu_4, 4)
    fontsize_theme = Theme(fontsize=20)
    set_theme!(fontsize_theme)
    fig = Figure(size = (1400, 600))
    
    # Create a subplot for the main plot
    ax = Axis(fig[1, 1], title = title, xticks = (1:70, networks_real_names[permsort]), 
              yticks = (0:1:6, [L"0", L"1", L"2", L"3", L"4", L"5", L"6"]),
              xticklabelrotation = pi/2, xticklabelsize = 12, xlabel = L"Desikan brain regions$$", ylabel = L"Shapley Value$$")
    
    barplot!(ax, shapvalues[permsort], color = color[end])
    errorbars!(ax, collect(1:70), shapvalues[permsort], std_values, whiskerwidth = 10, color = :black)

    display(fig)




    return fig
end
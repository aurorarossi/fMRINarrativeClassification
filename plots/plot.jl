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
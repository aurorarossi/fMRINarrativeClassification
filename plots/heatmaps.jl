using CairoMakie, NPZ

network= npzread("data/networks/sub-049_task-schema_run-1_space-MNI152_desc-mcf_bold_network_7networks.npy")

network = network[:,:,1:8]

colormap = :RdBu_10
# Create a Figure with 3 subplots
for i in 1:8
    if i == 1
        size1 = 850
        k =2
    else
        size1 = 800
        k = 1
    end
fig = CairoMakie.Figure(size = (size1,800))
ax = Axis(fig[1, k])
CairoMakie.heatmap!(ax, network[:,:,i], colormap=colormap, colorrange = (-1,1))
ax.yreversed = true
if i ==1
cbar = Colorbar(fig, label = "", width = 20,colormap=colormap, colorrange = (-1,1))
fig[1, 1] = cbar
end
display(fig)

CairoMakie.save("plots/plotspaperheatmap_$(i).png", fig)
end
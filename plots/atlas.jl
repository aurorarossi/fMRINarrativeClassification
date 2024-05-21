using CairoMakie, NPZ, Statistics

l_coords = npzread("data/L.pial.coords.npy")
r_coords = npzread("data/R.pial.coords.npy")
l_triangles = npzread("data/L.pial.triangles.npy")
r_triangles = npzread("data/R.pial.triangles.npy")
labels = npzread("data/labels_100.npy")


total_coords = vcat(l_coords, r_coords)
total_triangles = vcat(l_triangles .+ 1, r_triangles .+ 32493)


mapvalues = zeros(size(total_coords, 1))



for i in 1:100
    mapvalues[labels .== i] .= labels[i]
end



f =CairoMakie.Figure(resolution=(800, 800))
axes = []
meshes = []


camera_positions = [(-1, 0, 0), (0, -0.0001, 0.1), (1, 0, 0)]


    ax = f[1, 1] =CairoMakie.LScene(f, scenekw=(camera =CairoMakie.cam3d!, show_axis = false))
    ax.show_axis = false
    mesh = mesh!(ax, total_coords, total_triangles, color=labels,colormap = :RdBu_10)
    cam = cameracontrols(ax.scene)
    cam.eyeposition[] = camera_positions[2]
    push!(axes, ax)
    push!(meshes, mesh)

# save the figure

display(f)
CairoMakie.save("plots/plotspaper/atlas.png", f)
using CairoMakie, NPZ, LaTeXStrings

data = npzread("/data/coati/share/brain/new_project/timeseries/schema-run1-100/sub-049_task-schema_run-1_space-MNI152_desc-mcf_bold_timeseries_7networks.npy")

filtered  = npzread("/data/coati/share/brain/new_project/filtereddata.npy")
# plot the signal of one voxel

fontsize_theme = Theme(fontsize=30)
set_theme!(fontsize_theme)
color = categorical_colors(:RdBu_4,4)

fig = Figure(resolution = (1200, 600))
ax = Axis(fig[1, 1], xlabel = L"Time points$$", ylabel = L"Clean BOLD signal of a voxel$$",yticks = (-5:5:5, [L"-5", L"0", L"5"]), xticks = (0:200:400, [L"0", L"200", L"400"]))
lines!(ax, filtered[40,45,40,:], color = color[end] , linewidth = 2 , )


# save the plot

save("/home/aurossi/new_project/ZNEW/plots/plotspaper/onevoxelsignal.png", fig)


# two lines

fig = Figure(resolution = (1200, 600))
ax = Axis(fig[1, 1], xlabel = L"Time points$$", ylabel = L"BOLD signal of a ROI$$",yticks = (-2:1:2, [L"-2", L"-1", L"0", L"1", L"2"]), xticks = (0:200:400, [L"0", L"200", L"400"]))
lines!(ax, data[1,:], color = color[end] , linewidth = 2 , )
lines!(ax, data[2,:], color = color[1] , linewidth = 2 , )

# save the plot

save("/home/aurossi/new_project/ZNEW/plots/plotspaper/twoROIsignal.png", fig)


rawdata =npzread("/data/coati/share/brain/new_project/rawdata.npy")
fig = Figure(resolution = (1200, 600))
ax = Axis(fig[1, 1], xlabel = L"Time points$$", ylabel = L"Raw BOLD signal of a voxel$$", yticks = (300:10:330, [L"300", L"310", L"320", L"330"]), xticks = (0:200:400, [L"0", L"200", L"400"]))
lines!(ax, rawdata[40,45,40,:], color = color[end] , linewidth = 2 , )

# save the plot

save("/home/aurossi/new_project/ZNEW/plots/plotspaper/rawonevoxelsignal.png", fig)
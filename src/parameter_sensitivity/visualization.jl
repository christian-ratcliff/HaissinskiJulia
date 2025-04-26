# """
# visualization.jl - Sensitivity visualization tools

# This file implements visualization tools for sensitivity analysis results,
# including parameter scans and gradient information.
# """

# using Plots
# using LaTeXStrings


# """
#     plot_sensitivity_scan(
#         param_range, 
#         fom_values, 
#         gradient_values, 
#         gradient_errors;
#         param_name="Parameter", 
#         fom_name="Figure of Merit"
#     )

# Plot the results of a parameter scan with gradient information.
# """
# function plot_sensitivity_scan(
#     param_range, 
#     fom_values, 
#     gradient_values, 
#     gradient_errors;
#     param_name="Parameter", 
#     fom_name="Figure of Merit"
# )
#     # Create FoM plot
#     p1 = plot(param_range, fom_values, 
#              label="$fom_name", 
#              xlabel=param_name, 
#              ylabel=fom_name,
#              linewidth=2,
#              marker=:circle)
             
#     # Create gradient plot
#     p2 = plot(param_range, gradient_values, 
#              ribbon=gradient_errors,
#              label="d($fom_name)/d($param_name)", 
#              xlabel=param_name, 
#              ylabel="Gradient",
#              linewidth=2,
#              marker=:circle)
    
#     # Add a zero line on gradient plot
#     hline!(p2, [0], linestyle=:dash, color=:black, label=nothing)
             
#     # Combine plots
#     plot(p1, p2, layout=(2,1), size=(800, 600), dpi=300)
# end
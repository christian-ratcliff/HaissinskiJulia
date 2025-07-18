include("../src/StochasticHaissinski.jl")

begin
    using .StochasticHaissinski
    using StochasticAD
    using BenchmarkTools
    using Statistics
    using Random
    using LinearAlgebra
    using Distributions
    using StructArrays
    using CairoMakie
end


Random.seed!(1000)
# Set up base physical parameters
begin
    E0_ini = 4e9
    mass = MASS_ELECTRON
    voltage = 5e6
    harmonic = 360
    radius = 250.0
    pipe_radius = 0.00025
    α_c = 3.68e-4
    ϕs = 5π/6
    freq_rf = let
        γ = E0_ini/mass
        β = sqrt(1 - 1/γ^2)
        (ϕs + 10*π/180) * β * SPEED_LIGHT / (2π)
    end

    # Distribution parameters
    μ_z = 0.0
    μ_E = 0.0
    σ_E0 = 1e6
    σ_z0 = let
        γ = E0_ini/mass
        β = sqrt(1 - 1/γ^2)
        ω_rev = 2 * π / ((2*π*radius) / (β*SPEED_LIGHT))
        sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c*E0_ini/harmonic/voltage/abs(cos(ϕs))) * σ_E0 / E0_ini
    end
end;



begin
    println("="^60)
    println("Retyped Parametric StochasticAD Implementation")
    println("="^60)

    function longitudinal_evolve_parametric(
        particles_float64::StructArray{Particle{Float64}},
        E0_param::TE, 
        mass_param::TM, 
        voltage_param::TV, 
        harmonic_param::TH,
        radius_param::TR, 
        α_c_param::TA, 
        ϕs_param::TP, 
        freq_rf_param::TF,
        n_turns::Int
    ) where {TE, TM, TV, TH, TR, TA, TP, TF}
        
        # Determine the working type from physics parameters
        sample_computation = E0_param + voltage_param + radius_param + α_c_param + ϕs_param + freq_rf_param
        T = typeof(sample_computation)
        
        # Create zero of the working type T through arithmetic
        zero_T = sample_computation * 0
        
        # Copy Float64 particles to working type T using arithmetic operations
        n_particles = length(particles_float64)
        coords_T = Vector{Coordinate{T}}(undef, n_particles)
        
        for i in 1:n_particles
            # Use arithmetic to create T-typed values, avoiding type conversion
            z_T = zero_T + particles_float64.coordinates.z[i]
            ΔE_T = zero_T + particles_float64.coordinates.ΔE[i]
            coords_T[i] = Coordinate{T}(z_T, ΔE_T)
        end
        
        # Create StructArray with working type T
        particles = StructArray{Particle{T}}((StructArray(coords_T),))
        
        # Pre-compute physical constants in working type T
        γ0 = E0_param / mass_param
        β0 = sqrt(1 - 1/γ0^2)
        η0 = α_c_param - 1/γ0^2
        sin_ϕs = sin(ϕs_param)
        rf_factor = freq_rf_param * 2π / (β0 * SPEED_LIGHT)
        
        # Initial spreads - convert to working type through arithmetic
        σ_E0_initial = zero_T + std(particles_float64.coordinates.ΔE)
        
        # Main evolution loop - now all arrays can hold type T
        for turn in 1:n_turns
            # RF voltage kick
            for i in 1:n_particles
                ϕ_val = -particles.coordinates.z[i] * rf_factor + ϕs_param
                particles.coordinates.ΔE[i] += voltage_param * (sin(ϕ_val) - sin_ϕs)
            end
            
            # Quantum excitation (stochastic effect)
            ∂U_∂E = 4 * 8.85e-5 * (E0_param/1e9)^3 / radius_param
            excitation = sqrt(1-(1-∂U_∂E)^2) * σ_E0_initial
            for i in 1:n_particles
                # randn() gives Float64, arithmetic with excitation promotes to T
                particles.coordinates.ΔE[i] += excitation * randn()
            end
            
            # Synchrotron radiation damping
            damping_factor = 1 - ∂U_∂E
            for i in 1:n_particles
                particles.coordinates.ΔE[i] *= damping_factor
            end
            
            # Update reference energy
            E0_param = E0_param - 4 * 8.85e-5 * (E0_param/1e9)^3 / radius_param * E0_param / 4
            γ0 = E0_param / mass_param
            β0 = sqrt(1 - 1/γ0^2)
            η0 = α_c_param - 1/γ0^2
            
            # Update phase advance
            coeff = 2π * harmonic_param * η0 / (β0^2 * E0_param)
            for i in 1:n_particles
                ϕ_i = -(particles.coordinates.z[i] * rf_factor - ϕs_param)
                ϕ_i += coeff * particles.coordinates.ΔE[i]
                particles.coordinates.z[i] = (-ϕ_i + ϕs_param) / rf_factor
            end
            
            # Update RF factor
            rf_factor = freq_rf_param * 2π / (β0 * SPEED_LIGHT)
        end
        
        # Return final energy spread in MeV
        σ_E_final = std(particles.coordinates.ΔE)
        return σ_E_final / 1e6
    end

    function voltage_sensitivity(voltage_scale; n_particles=1000, n_turns=200)
        particles, _, _, _ = generate_particles(
            μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf
        )
        
        voltage_scaled = voltage_scale * voltage
        
        return longitudinal_evolve_parametric(
            particles, E0_ini, mass, voltage_scaled, harmonic,
            radius, α_c, ϕs, freq_rf, n_turns
        )
    end

    function energy_sensitivity(energy_scale; n_particles=1000, n_turns=200)
        particles, _, _, _ = generate_particles(
            μ_z, μ_E, σ_z0, σ_E0, n_particles, E0_ini, mass, ϕs, freq_rf
        )
        
        E0_scaled = energy_scale * E0_ini
        
        return longitudinal_evolve_parametric(
            particles, E0_scaled, mass, voltage, harmonic,
            radius, α_c, ϕs, freq_rf, n_turns
        )
    end
end;

# Test basic functionality
nparts = 1000;
nturn = 4000;
begin
    
    
    println("Testing retyped parametric beam evolution...")
    test_voltage = voltage_sensitivity(1.0; n_particles=nparts, n_turns=nturn)
    test_energy = energy_sensitivity(1.0; n_particles=nparts, n_turns=nturn)
    
    println("Voltage sensitivity test: $(round(test_voltage, digits=4)) MeV")
    println("Energy sensitivity test: $(round(test_energy, digits=4)) MeV\n")
end

# Finite difference implementation
begin
    println("="^60)
    println("FINITE DIFFERENCE DERIVATIVES")
    println("="^60)

    function finite_difference_scan(f, x; h=5e-3, n_samples=100)
        step_sizes = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        
        # println("Step size scan:")
        # for h in step_sizes
        #     deriv = (f(x + h) - f(x - h)) / (2 * h)
        #     println("  h = $h: derivative = $(round(deriv, digits=4))")
        # end
        
        
        # Use h = 5e-3 for actual calculation
        # h = 5e-3
        derivatives = Float64[]
        for i in 1:n_samples
            deriv = (f(x + h) - f(x - h)) / (2 * h)
            push!(derivatives, deriv)
        end
        
        for hh in step_sizes
            derivatives1 = Float64[]
            for i in 1:n_samples
                deriv = (f(x + hh) - f(x - hh)) / (2 * hh)
                push!(derivatives1, deriv)
            end
            println("  h = $hh: derivative = $(round(mean(derivatives1), digits=4)) ± $(round(std(derivatives1) / sqrt(n_samples), digits=4))")
        end
        
        return mean(derivatives), std(derivatives) / sqrt(n_samples)
    end
    n_samples = 500
    h = 5e-3
    # Voltage derivative
    println("\n1. Voltage sensitivity:")
    fd_voltage_mean, fd_voltage_err = finite_difference_scan(
        x -> voltage_sensitivity(x; n_particles=nparts, n_turns=nturn), 1.0; h=h,  n_samples = n_samples
    )
    println("   Final result (h=$(h), $(n_samples) samples): $(round(fd_voltage_mean, digits=4)) ± $(round(fd_voltage_err, digits=6))")

    # Energy derivative  
    println("\n2. Energy sensitivity:")
    fd_energy_mean, fd_energy_err = finite_difference_scan(
        x -> energy_sensitivity(x; n_particles=nparts, n_turns=nturn), 1.0; h=h,  n_samples = n_samples
    )
    println("   Final result (h=$(h), $(n_samples) samples): $(round(fd_energy_mean, digits=4)) ± $(round(fd_energy_err, digits=6))")
end

# StochasticAD implementation
begin
    println("\n" * "="^60)
    println("STOCHASTIC AD DERIVATIVES")
    println("="^60)

    function stochastic_ad_derivative(f; n_samples=10)
        derivatives = Float64[]
        
        for i in 1:n_samples
            deriv = derivative_estimate(f, 1.0)
            push!(derivatives, deriv)
        end
        
        return mean(derivatives), std(derivatives) / sqrt(n_samples)
    end
    
    n_samples = 50
    # Voltage derivative with StochasticAD
    println("\n1. Voltage sensitivity:")
    ad_voltage_mean, ad_voltage_err = stochastic_ad_derivative(
        x -> voltage_sensitivity(x; n_particles=nparts, n_turns=nturn); n_samples = n_samples
    )
    println("   d(σ_E)/d(voltage_scale) = $ad_voltage_mean ± $ad_voltage_err")

    # Energy derivative with StochasticAD  
    println("\n2. Energy sensitivity:")
    ad_energy_mean, ad_energy_err = stochastic_ad_derivative(
        x -> energy_sensitivity(x; n_particles=nparts, n_turns=nturn)
    )
    println("   d(σ_E)/d(energy_scale) = $ad_energy_mean ± $ad_energy_err")
end

# Comparison
begin
    println("="^60)
    println("ERFORMANCE COMPARISON")
    println("="^60)

    # Parameters for comparison
    nparts = 1000
    nturn = 4000
    h = 5e-3
    fd_samples_num = 500
    ad_samples_num = 10
    # Realistic finite difference timing (100 samples for reliable estimate)
    println("Timing finite difference ($(fd_samples_num) samples, h=$(h)):")
    
    # Voltage derivative
    fd_voltage_time = @elapsed begin
        

        fd_samples = Float64[]
        for i in 1:fd_samples_num
            deriv = (voltage_sensitivity(1.0 + h; n_particles=nparts, n_turns=nturn) - 
                    voltage_sensitivity(1.0 - h; n_particles=nparts, n_turns=nturn)) / (2 * h)
            push!(fd_samples, deriv)
        end
        fd_voltage_mean = mean(fd_samples)
        fd_voltage_err = std(fd_samples) / sqrt(fd_samples_num)
    end
    
    # Energy derivative
    fd_energy_time = @elapsed begin
        # h = 5e-3
        fd_samples = Float64[]
        for i in 1:fd_samples_num
            deriv = (energy_sensitivity(1.0 + h; n_particles=nparts, n_turns=nturn) - 
                    energy_sensitivity(1.0 - h; n_particles=nparts, n_turns=nturn)) / (2 * h)
            push!(fd_samples, deriv)
        end
        fd_energy_mean = mean(fd_samples)
        fd_energy_err = std(fd_samples) / sqrt(fd_samples_num)
    end
    
    fd_total_time = fd_voltage_time + fd_energy_time
    println("  Voltage: $(round(fd_voltage_time, digits=2))s → $(round(fd_voltage_mean, digits=4)) ± $(round(fd_voltage_err, digits=4))")
    println("  Energy: $(round(fd_energy_time, digits=2))s → $(round(fd_energy_mean, digits=4)) ± $(round(fd_energy_err, digits=4))")
    println("  Total time: $(round(fd_total_time, digits=2))s")
    println("  Function evaluations: 400 (200 each)")

    println("\nTiming StochasticAD (10 samples):")
    
    # ad_samples_num = 10
    # Voltage derivative
    ad_voltage_time = @elapsed begin
        ad_samples = Float64[]
        for i in 1:ad_samples_num
            deriv = derivative_estimate(x -> voltage_sensitivity(x; n_particles=nparts, n_turns=nturn), 1.0)
            push!(ad_samples, deriv)
        end
        ad_voltage_mean = mean(ad_samples)
        ad_voltage_err = std(ad_samples) / sqrt(ad_samples_num)
    end
    
    # Energy derivative
    ad_energy_time = @elapsed begin
        ad_samples = Float64[]
        for i in 1:ad_samples_num
            deriv = derivative_estimate(x -> energy_sensitivity(x; n_particles=nparts, n_turns=nturn), 1.0)
            push!(ad_samples, deriv)
        end
        ad_energy_mean = mean(ad_samples)
        ad_energy_err = std(ad_samples) / sqrt(ad_samples_num)
    end
    
    ad_total_time = ad_voltage_time + ad_energy_time
    println("  Voltage: $(round(ad_voltage_time, digits=2))s → $(round(ad_voltage_mean, digits=4)) ± $(round(ad_voltage_err, digits=4))")
    println("  Energy: $(round(ad_energy_time, digits=2))s → $(round(ad_energy_mean, digits=4)) ± $(round(ad_energy_err, digits=4))")
    println("  Total time: $(round(ad_total_time, digits=2))s")
    println("  Function evaluations: 200 (10 each)")

    println("\nOverall speedup: $(round(fd_total_time/ad_total_time, digits=2))x")
    # println("Evaluation efficiency: $(round(400/20, digits=1))x fewer function calls")
end

# Detailed Comparison Table
begin
    println("\n" * "="^60)
    println("DETAILED COMPARISON")
    println("="^60)
    println("\n Parameter          | Finite Diff    | StochasticAD")
    println("-------------------|----------------|------------------------")
    println(" Voltage scale     | $(round(fd_voltage_mean, digits=3)) ± $(round(fd_voltage_err, digits=3)) | $(round(ad_voltage_mean, digits=3)) ± $(round(ad_voltage_err, digits=3))")
    println(" Energy scale      | $(round(fd_energy_mean, digits=3)) ± $(round(fd_energy_err, digits=3)) | $(round(ad_energy_mean, digits=3)) ± $(round(ad_energy_err, digits=3))")
    
    # Calculate relative differences
    voltage_diff = abs(fd_voltage_mean - ad_voltage_mean) / abs(fd_voltage_mean) * 100
    energy_diff = abs(fd_energy_mean - ad_energy_mean) / abs(fd_energy_mean) * 100
    
    println("\n Relative differences:")
    println(" Voltage: $(round(voltage_diff, digits=1))%")
    println(" Energy:  $(round(energy_diff, digits=1))%")
    
    # Statistical significance
    voltage_combined_error = sqrt(fd_voltage_err^2 + ad_voltage_err^2)
    energy_combined_error = sqrt(fd_energy_err^2 + ad_energy_err^2)
    
    voltage_sigma_diff = abs(fd_voltage_mean - ad_voltage_mean) / voltage_combined_error
    energy_sigma_diff = abs(fd_energy_mean - ad_energy_mean) / energy_combined_error
    
    println("\n Statistical significance:")
    println(" Voltage: $(round(voltage_sigma_diff, digits=2))σ difference")
    println(" Energy:  $(round(energy_sigma_diff, digits=2))σ difference")
    
    # Agreement assessment
    if voltage_sigma_diff < 2.0 && energy_sigma_diff < 2.0
        println("\n Good agreement (both within 2σ)")
    elseif voltage_sigma_diff < 3.0 && energy_sigma_diff < 3.0
        println("\n Reasonable agreement (both within 3σ)")
    else
        println("\n Methods disagree beyond statistical uncertainty")
    end
    
    # Performance summary
    println("\n PERFORMANCE SUMMARY:")
    println(" Method           | Time [s] | Evaluations | Accuracy")
    println("------------------|----------|-------------|----------")
    println(" Finite Diff      | $(lpad(round(fd_total_time, digits=1), 8)) | $(lpad(400, 11)) | ±$(round(mean([fd_voltage_err, fd_energy_err]), digits=3))")
    println(" StochasticAD     | $(lpad(round(ad_total_time, digits=1), 8)) | $(lpad(20, 11)) | ±$(round(mean([ad_voltage_err, ad_energy_err]), digits=3))")
    println(" Improvement      | $(lpad(round(fd_total_time/ad_total_time, digits=1), 8))x | $(lpad(round(400/20, digits=1), 11))x | $(round(mean([fd_voltage_err, fd_energy_err])/mean([ad_voltage_err, ad_energy_err]), digits=1))x better")
end


# Parameter scan visualization for both voltage and energy
begin
    println("\n" * "="^60)
    println("PARAMETER SCAN VISUALIZATION")
    println("="^60)

    scales = range(0.999, 1.001, length=11)

    # Voltage parameter scan
    # voltage_scales = range(0.99, 1.01, length=11)
    σ_E_voltage_values = Float64[]
    σ_E_voltage_std = Float64[]
    n_samples = 100
    println("\nScanning voltage parameter...")
    # for scale in voltage_scales
    for scale in scales
        σ_E_runs = [voltage_sensitivity(scale; n_particles=nparts, n_turns=nturn) for _ in 1:n_samples]
        σ_E = mean(σ_E_runs)
        std_σ_E = std(σ_E_runs) / sqrt(n_samples)
        push!(σ_E_voltage_values, σ_E)
        push!(σ_E_voltage_std, std_σ_E)
        println("  Scale: $scale, σ_E: $(round(σ_E, digits=4)) ± $(round(std_σ_E, digits=4)) MeV")
    end

    # Energy parameter scan
    # energy_scales = range(0.99, 1.01, length=11)
    σ_E_energy_values = Float64[]
    σ_E_energy_std = Float64[]


    println("\nScanning energy parameter...")
    # for scale in energy_scales
    for scale in scales
        σ_E_runs = [energy_sensitivity(scale; n_particles=nparts, n_turns=nturn) for _ in 1:100]
        σ_E = mean(σ_E_runs)
        std_σ_E = std(σ_E_runs) / sqrt(n_samples)
        push!(σ_E_energy_values, σ_E)
        push!(σ_E_energy_std , std_σ_E)
        println("  Scale: $scale, σ_E: $(round(σ_E, digits=4)) ± $(round(std_σ_E, digits=4)) MeV")
    end

    # Create figure with both plots
    fig = Figure(size = (1200, 500))
    
    # Voltage sensitivity plot
    ax1 = Axis(fig[1, 1], 
            title = "Energy Spread Sensitivity to Voltage",
            xlabel = "Voltage Scale Factor", 
            ylabel = "Final σ_E [MeV]")
    errorbars!(ax1, scales, σ_E_voltage_values, σ_E_voltage_std; whiskerwidth = 10, color= :black, label = "Simulation" )
    # scatter!(ax1, voltage_scales, σ_E_voltage_values, markersize = 12, color = :blue, label = "Simulation")
    scatter!(ax1, scales, σ_E_voltage_values, markersize = 12, color = :blue, label = "Simulation")

    # Add tangent lines for voltage
    # nominal_idx_v = findfirst(x -> x ≈ 1.0, voltage_scales)
    nominal_idx_v = findfirst(x -> x ≈ 1.0, scales)

    nominal_σE_v = σ_E_voltage_values[nominal_idx_v]

    plot_range = range(0.999, 1.001, length=50)
    # x_range_v = range(0.99, 1.01, length=50)
    x_range_v = plot_range
    fd_tangent_v = nominal_σE_v .+ fd_voltage_mean .* (x_range_v .- 1.0)
    lines!(ax1, x_range_v, fd_tangent_v, linestyle = :dash, color = :red, linewidth = 3,
        label = "FD ($(round(fd_voltage_mean, digits=3)) ± $(round(fd_voltage_err, digits=3)))")

    ad_tangent_v = nominal_σE_v .+ ad_voltage_mean .* (x_range_v .- 1.0)
    lines!(ax1, x_range_v, ad_tangent_v, linestyle = :dash, color = :green, linewidth = 3,
        label = "AD ($(round(ad_voltage_mean, digits=3)) ± $(round(ad_voltage_err, digits=3)))")

    # axislegend(ax1, position = :lt)

    # Energy sensitivity plot
    ax2 = Axis(fig[1, 2], 
            title = "Energy Spread Sensitivity to Energy",
            xlabel = "Energy Scale Factor", 
            ylabel = "Final σ_E [MeV]")
            
    errorbars!(ax2, scales, σ_E_energy_values, σ_E_energy_std; whiskerwidth = 10, color= :black, label = "Simulation" )

    # scatter!(ax2, energy_scales, σ_E_energy_values, markersize = 12, color = :red, label = "Simulation")
    scatter!(ax2, scales, σ_E_energy_values, markersize = 12, color = :red, label = "Simulation")

    # Add tangent lines for energy
    # nominal_idx_e = findfirst(x -> x ≈ 1.0, energy_scales)
    nominal_idx_e = findfirst(x -> x ≈ 1.0, scales)

    nominal_σE_e = σ_E_energy_values[nominal_idx_e]

    # x_range_e = range(0.99, 1.01, length=50)
    x_range_e = plot_range
    fd_tangent_e = nominal_σE_e .+ fd_energy_mean .* (x_range_e .- 1.0)
    lines!(ax2, x_range_e, fd_tangent_e, linestyle = :dash, color = :red, linewidth = 3,
        label = "FD ($(round(fd_energy_mean, digits=3)) ± $(round(fd_energy_err, digits=3)))")

    ad_tangent_e = nominal_σE_e .+ ad_energy_mean .* (x_range_e .- 1.0)
    lines!(ax2, x_range_e, ad_tangent_e, linestyle = :dash, color = :green, linewidth = 3,
        label = "AD ($(round(ad_energy_mean, digits=3)) ± $(round(ad_energy_err, digits=3)))")

    axislegend(ax1, merge=true, position = :lt)
    axislegend(ax2, merge=true, position = :lt)
    display(fig)
    
    # Compute prediction accuracy for both
    println("\nTangent line prediction accuracy:")
    
    # Voltage prediction errors
    prediction_errors_fd_v = Float64[]
    prediction_errors_ad_v = Float64[]
    # for (i, scale) in enumerate(voltage_scales)
    for (i, scale) in enumerate(scales)

        if scale != 1.0
            predicted_fd = nominal_σE_v + fd_voltage_mean * (scale - 1.0)
            predicted_ad = nominal_σE_v + ad_voltage_mean * (scale - 1.0)
            actual = σ_E_voltage_values[i]
            push!(prediction_errors_fd_v, abs(predicted_fd - actual))
            push!(prediction_errors_ad_v, abs(predicted_ad - actual))
        end
    end
    
    # Energy prediction errors
    prediction_errors_fd_e = Float64[]
    prediction_errors_ad_e = Float64[]
    # for (i, scale) in enumerate(energy_scales)
    for (i, scale) in enumerate(scales)

        if scale != 1.0
            predicted_fd = nominal_σE_e + fd_energy_mean * (scale - 1.0)
            predicted_ad = nominal_σE_e + ad_energy_mean * (scale - 1.0)
            actual = σ_E_energy_values[i]
            push!(prediction_errors_fd_e, abs(predicted_fd - actual))
            push!(prediction_errors_ad_e, abs(predicted_ad - actual))
        end
    end
    
    println("  Voltage - FD RMS error: $(round(sqrt(mean(prediction_errors_fd_v.^2)), digits=4)) MeV")
    println("  Voltage - AD RMS error: $(round(sqrt(mean(prediction_errors_ad_v.^2)), digits=4)) MeV")
    println("  Energy - FD RMS error: $(round(sqrt(mean(prediction_errors_fd_e.^2)), digits=4)) MeV")
    println("  Energy - AD RMS error: $(round(sqrt(mean(prediction_errors_ad_e.^2)), digits=4)) MeV")
end


# Final particle distribution visualization
begin
    println("\n" * "="^60)
    println("FINAL PARTICLE DISTRIBUTION")
    println("="^60)

    # Run one final simulation to get particle distributions
    final_particles, _, _, _ = generate_particles(
        μ_z, μ_E, σ_z0, σ_E0, 2000, E0_ini, mass, ϕs, freq_rf
    )
    
    # Evolve particles
    voltage_scaled = 1.0 * voltage
    final_result = longitudinal_evolve_parametric(
        final_particles, E0_ini, mass, voltage_scaled, harmonic,
        radius, α_c, ϕs, freq_rf, 1000
    )
    
    # Create figure with subplots
    fig = Figure(size = (1000, 400))
    
    # Longitudinal distribution
    ax1 = Axis(fig[1, 1], 
            title = "Final Longitudinal Distribution",
            xlabel = "z [m]", 
            ylabel = "Count")
    hist!(ax1, final_particles.coordinates.z, bins = 50, color = (:blue, 0.7))
    
    # Energy distribution  
    ax2 = Axis(fig[1, 2],
            title = "Final Energy Distribution", 
            xlabel = "ΔE [MeV]", 
            ylabel = "Count")
    hist!(ax2, final_particles.coordinates.ΔE ./ 1e6, bins = 50, color = (:red, 0.7))
    
    display(fig)
    
    println("Final σ_z: $(round(std(final_particles.coordinates.z)*1000, digits=2)) mm")
    println("Final σ_E: $(round(std(final_particles.coordinates.ΔE)/1e6, digits=3)) MeV")
end



begin
    println("\n" * "="^60)
    println("IMPLEMENTATION COMPLETE")
    println("="^60)
end

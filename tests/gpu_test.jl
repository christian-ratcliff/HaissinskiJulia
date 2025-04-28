# # # Save as gpu_test.jl
# using CUDA

# function test_gpu()
#     if CUDA.functional()
#         println("CUDA is functional!")
#         println("Device: $(CUDA.name(CUDA.device()))")
        
#         # Try a simple kernel
#         a = CUDA.ones(1024)
#         b = CUDA.ones(1024)
#         c = a + b
#         println("Simple kernel test passed: sum = $(sum(Array(c)))")
#     else
#         println("CUDA is not functional!")
#     end
# end

# test_gpu()


# # Save as cuda_diagnostic.jl
# # println("Starting CUDA diagnostics...")

# # # Version information
# # println("Julia version: ", VERSION)

# # # Try to load CUDA and report errors
# # try
# #     println("Attempting to load CUDA module...")
# #     @time using CUDA
# #     println("CUDA module loaded successfully!")
    
# #     # Check if CUDA is functional
# #     if CUDA.functional()
# #         println("CUDA is FUNCTIONAL")
# #         println("Device: ", CUDA.name(CUDA.device()))
# #         println("Compute capability: ", CUDA.capability(CUDA.device()))
# #         println("CUDA driver version: ", CUDA.driver_version())
# #     else
# #         println("CUDA loaded but NOT FUNCTIONAL")
# #     end
# # catch e
# #     println("Error loading CUDA: ", e)
# # end

# # # Check system CUDA detection
# # println("\nChecking system CUDA status...")
# # try
# #     run(`nvidia-smi`)
# # catch e
# #     println("Failed to run nvidia-smi: ", e)
# # end

using CUDA

println("CUDA_VISIBLE_DEVICES: ", get(ENV, "CUDA_VISIBLE_DEVICES", "not set"))
println("GPU devices: ", get(ENV, "SLURM_GPUS", "not set"))
CUDA.device!(0)  # Explicitly select first GPU
println("CUDA device: ", CUDA.name(CUDA.device()))
@show CUDA.functional()
@show CUDA.driver_version()
@show CUDA.runtime_version()  # Instead of version()
@show CUDA.capability(CUDA.device())  # Instead of capability()


include("../SolutionFunctions.jl")
using Plots
using Statistics

# Parameters
P = 50.0                 # Choose p so that there are at least three bound states
N = 500                  # number of interior grid points
q = q_of_p(P)
L = 20.0 / q             # domain size
h = L / (N + 1)          # grid spacing
eta = 0.1             # perturbation amplitude (increased to see Rabi oscillations)

# Step A: Solve the TISE to get eigenstates and eigenvalues
println("Step A: Solving TISE for static potential...")
V_static(xi) = -sech(q * xi)^2
ws = SolverWorkspace(N)
E, U, xi = solve_static_schrodinger(L, h, V_static, ws)

# Store the eigenvalues we need
epsilon_0 = E[1]
epsilon_2 = E[3]
Omega = epsilon_2 - epsilon_0  # resonant frequency

println("Found $(length(E)) bound states")
println("ε₀ = $(epsilon_0)")
println("ε₂ = $(epsilon_2)")
println("Ω = ε₂ - ε₀ = $(Omega)")

# Define time-dependent potential
V_dynamic(xi, tau) = V_static(xi) * (1.0 + eta * sin(Omega * tau))

# Step B: Set initial condition - start in ground state u₀
println("\nStep B: Setting initial condition Ψ(ξ,0) = u₀(ξ)...")

# Normalise all states using L2 norm
U_norm = similar(U)
for i in 1:size(U, 2)
    U_norm[:, i] = normalise_L2(U[:, i], h)
end

psi = ComplexF64.(U_norm[:, 1])  # ground state eigenvector

# Step C: Evolve using Crank-Nicolson
println("\nStep C: Evolving wavefunction with time-dependent potential (LONG TIME)...")
dt = 0.1                  # time step
# Increase simulation time to improve FFT resolution and capture multiple Rabi cycles
T_final = 1024π/Omega              
nsteps = round(Int, T_final / dt)

println("T_final = $T_final (approx $(round(T_final/(2π/Omega), digits=1)) periods)")

# Evolve and project onto basis at each step
# Optimization: only project onto the first few states to save time
n_proj = min(10, size(U_norm, 2))
basis_proj = U_norm[:, 1:n_proj]

psi_evolved, coeffs_history = evolve_dynamic_coeffs!(
    psi, L, h, V_dynamic, dt;
    nsteps=nsteps,
    ws=ws,
    basis=basis_proj,
    store=true
)

# Step D: Analyze coefficients
println("\nStep D: Analyzing state coefficients...")
coeffs_matrix = hcat(coeffs_history...)  # shape: (n_proj, n_timesteps)
magnitudes = abs.(coeffs_matrix)         # |cₙ(τ)|

# Time array scaled by Ω/2π
times = range(0, T_final, length=nsteps+1)
t_scaled = times .* Omega ./ (2 * pi)

# Plot magnitudes of first few states
plt_mag = plot(t_scaled, magnitudes[1, :], label="n=0 (ground)", lw=2, 
     xlabel="tΩ/(2π)", ylabel="|cₙ|", title="State Magnitudes vs Scaled Time (Long Run)", dpi=300)
plot!(plt_mag, t_scaled, magnitudes[2, :], label="n=1", lw=2)
plot!(plt_mag, t_scaled, magnitudes[3, :], label="n=2 (resonant)", lw=2)
if size(magnitudes, 1) >= 4
    plot!(plt_mag, t_scaled, magnitudes[4, :], label="n=3", lw=2)
end
savefig(plt_mag, "Q4/state_magnitudes_long.png")
println("Saved Q4/state_magnitudes_long.png")

# Perform FFT on coefficients of ground and second excited states
# No zero padding, just raw FFT on the longer signal
f, S = discrete_fft(coeffs_matrix[1, :], dt)
f2, S2 = discrete_fft(coeffs_matrix[3, :], dt)

plt_fft = plot(f, abs.(S), label="n=0 (f ≈ |ε₀|/2π)", xlabel="Frequency f", ylabel="|FFT|", 
     title="FFT of Coefficients (Long Simulation)", xlim=(0, 0.25), lw=2, dpi=300)
plot!(plt_fft, f2, abs.(S2), label="n=2 (f ≈ |ε₂|/2π)", lw=2)

# Annotate peaks and drive frequency
annotate!(plt_fft, 0.14, 0.4, text("f₀ ≈ |ε₀|/2π", :blue, 10, :left))
annotate!(plt_fft, 0.07, 0.4, text("f₂ ≈ |ε₂|/2π", :orange, 10, :right))

savefig(plt_fft, "Q4/fft_coefficients_long.png")
println("Saved Q4/fft_coefficients_long.png")

# --- FFT of Probabilities ---
println("Computing FFT of probabilities...")
probs_0 = abs2.(coeffs_matrix[1, :])
probs_2 = abs2.(coeffs_matrix[3, :])

# Subtract mean to remove DC component and see the oscillation peak
probs_0_ac = probs_0 .- mean(probs_0)
probs_2_ac = probs_2 .- mean(probs_2)

fp, Sp0 = discrete_fft(probs_0_ac, dt, norm=:unitary)
fp2, Sp2 = discrete_fft(probs_2_ac, dt, norm=:unitary)

# Estimate Rabi frequency
# Calculate exact matrix element <2|V_static|0>
# V_pert(t) = eta * V_static * sin(Omega*t)
# Rabi frequency Omega_R = |<2| eta * V_static |0>|
V_vals = V_static.(xi)
# U_norm[:, 1] is ground state (n=0), U_norm[:, 3] is second excited state (n=2)
matrix_element = sum(conj(U_norm[:, 3]) .* V_vals .* U_norm[:, 1]) * h
V_20_exact = abs(matrix_element)
println("Calculated matrix element |<2|V|0>| = $V_20_exact")

Omega_R = eta * V_20_exact
f_Rabi = Omega_R / (2π)
println("Estimated Rabi frequency f_Rabi ≈ $f_Rabi")

plt_prob_fft = plot(fp, abs.(Sp0), label="FFT(|c₀|² - mean)", xlabel="Frequency f", ylabel="Magnitude",
                    title="FFT of State Probabilities (Rabi Oscillation)", xlim=(0, 0.01), lw=2, dpi=300)
plot!(plt_prob_fft, fp2, abs.(Sp2), label="FFT(|c₂|² - mean)", lw=2, ls=:dash)
vline!([f_Rabi], ls=:dot, color=:green, label="Est. Rabi Freq")

savefig(plt_prob_fft, "Q4/fft_probabilities_long.png")
println("Saved Q4/fft_probabilities_long.png")

# --- Zoomed in Plot for Splitting ---
println("Generating zoomed-in plot for Rabi splitting...")
f0_center = abs(epsilon_0) / (2π)
f2_center = abs(epsilon_2) / (2π)
span = 0.003  # Narrow window to see the splitting

# Use zero-padding to increase FFT resolution (interpolation)
pad_factor = 10
n_pad = length(coeffs_matrix[1, :]) * pad_factor
coeffs_0_padded = [coeffs_matrix[1, :]; zeros(ComplexF64, n_pad)]
coeffs_2_padded = [coeffs_matrix[3, :]; zeros(ComplexF64, n_pad)]

f_pad, S_pad0 = discrete_fft(coeffs_0_padded, dt)
f_pad2, S_pad2 = discrete_fft(coeffs_2_padded, dt)

# Find actual peaks in the window of interest for n=0
idx_window_0 = findall(x -> f0_center - span <= x <= f0_center + span, f_pad)
f_window_0 = f_pad[idx_window_0]
S_window_0 = abs.(S_pad0[idx_window_0])

# Find the two highest peaks
# Simple peak finding: find local maxima
peaks_0_indices = []
for i in 2:length(S_window_0)-1
    if S_window_0[i] > S_window_0[i-1] && S_window_0[i] > S_window_0[i+1]
        push!(peaks_0_indices, i)
    end
end
# Sort by magnitude and take top 2
sort!(peaks_0_indices, by = i -> S_window_0[i], rev=true)
top_peaks_0 = f_window_0[peaks_0_indices[1:2]]
observed_splitting_0 = abs(top_peaks_0[1] - top_peaks_0[2])

println("Observed Splitting (n=0): $observed_splitting_0")
println("Theoretical Splitting: $f_Rabi")

split_str = "Δf ≈ Ω_R / 2π"

p1 = plot(f_pad, abs.(S_pad0), title="n=0 Splitting (Ground)", xlabel="Frequency", ylabel="|FFT|", 
          xlim=(f0_center - span, f0_center + span), legend=false, lw=2)
# Mark the OBSERVED peaks
vline!(p1, top_peaks_0, ls=:dot, c=:red, label="Observed")
annotate!(p1, f0_center, maximum(S_window_0)*0.6, text(split_str, :black, 10))


# Find actual peaks for n=2
idx_window_2 = findall(x -> f2_center - span <= x <= f2_center + span, f_pad2)
f_window_2 = f_pad2[idx_window_2]
S_window_2 = abs.(S_pad2[idx_window_2])

peaks_2_indices = []
for i in 2:length(S_window_2)-1
    if S_window_2[i] > S_window_2[i-1] && S_window_2[i] > S_window_2[i+1]
        push!(peaks_2_indices, i)
    end
end
sort!(peaks_2_indices, by = i -> S_window_2[i], rev=true)
top_peaks_2 = f_window_2[peaks_2_indices[1:2]]
observed_splitting_2 = abs(top_peaks_2[1] - top_peaks_2[2])
split_str_2 = "Δf ≈ Ω_R / 2π"

p2 = plot(f_pad2, abs.(S_pad2), title="n=2 Splitting (Resonant)", xlabel="Frequency", ylabel="|FFT|", 
          xlim=(f2_center - span, f2_center + span), legend=false, lw=2, color=:orange)
vline!(p2, top_peaks_2, ls=:dot, c=:red)
annotate!(p2, f2_center, maximum(S_window_2)*0.6, text(split_str_2, :black, 10))

plt_zoom = plot(p1, p2, layout=(1, 2), size=(1000, 400), dpi=300, margin=5Plots.mm)
savefig(plt_zoom, "Q4/fft_splitting_zoom.png")
println("Saved Q4/fft_splitting_zoom.png")

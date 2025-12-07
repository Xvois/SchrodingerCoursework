include("../SolutionFunctions.jl")
using Plots

# Parameters
P = 30.0                 # Choose p so that there are at least three bound states
h = 0.05                 # spatial step size
q = q_of_p(P)
L = 15.0 / q             # domain size
N = round(Int, L / h) - 1
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

# Apply Hanning window and compute FFTs for coefficients
win = hann_window(length(coeffs_matrix[1, :]))
coeffs_0_windowed = coeffs_matrix[1, :] .* win
coeffs_2_windowed = coeffs_matrix[3, :] .* win

f, S = discrete_fft(coeffs_0_windowed, dt)
f2, S2 = discrete_fft(coeffs_2_windowed, dt)

# --- FFT of Probabilities ---
println("Computing FFT of probabilities...")
probs_0 = abs2.(coeffs_matrix[1, :])
probs_2 = abs2.(coeffs_matrix[3, :])

# Subtract mean to remove DC component and see the oscillation peak
probs_0_ac = probs_0 .- mean_value(probs_0)
probs_2_ac = probs_2 .- mean_value(probs_2)

# Apply window to probabilities too
probs_0_ac_win = probs_0_ac .* win
probs_2_ac_win = probs_2_ac .* win

fp, Sp0 = discrete_fft(probs_0_ac_win, dt, norm=:none)
fp2, Sp2 = discrete_fft(probs_2_ac_win, dt, norm=:none)

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

# Prepare FFT of probabilities (we will only show the zoomed splitting below)
plt_prob_fft = nothing

# --- Zoomed in Plot for Splitting ---
println("Generating zoomed-in plot for Rabi splitting...")
f0_center = abs(epsilon_0) / (2π)
f2_center = abs(epsilon_2) / (2π)
span = 0.003  # Wider window to show all features including the third peak

# Use zero-padding to increase FFT resolution (interpolation)
# Apply window BEFORE padding
pad_factor = 2
n_pad = length(coeffs_matrix[1, :]) * pad_factor
coeffs_0_padded = [coeffs_0_windowed; zeros(ComplexF64, n_pad)]
coeffs_2_padded = [coeffs_2_windowed; zeros(ComplexF64, n_pad)]

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

p1 = plot(f_pad, abs.(S_pad0), xlabel="Frequency", ylabel=math_label("|FFT|"),
          xlim=(f0_center - span, f0_center + span), legend=false, lw=2,
          titlefont=font(22, "Computer Modern"), guidefont=font(16), tickfont=font(12), fontfamily="Computer Modern",
          size=(620, 350), dpi=500, margin=6Plots.mm)
# Mark the OBSERVED peaks
vline!(p1, top_peaks_0, ls=:dot, c=:red, label="Observed")

# Annotate splitting with Delta f label
y_arrow = maximum(S_window_0) * 0.8
plot!(p1, [top_peaks_0[1], top_peaks_0[2]], [y_arrow, y_arrow], arrow=:both, color=:black, lw=1.5)
annotate!(p1, (top_peaks_0[1] + top_peaks_0[2])/2, y_arrow * 1.15, text(math_label("\\Delta f"), 14, :bottom))

# Save with consistent figure sizing
savefig(p1, "Q4/fft_splitting_zoom.png")
println("Saved Q4/fft_splitting_zoom.png")
